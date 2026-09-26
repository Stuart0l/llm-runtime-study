"""Request-facing caches and the runtime allocator over a paged store."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from mini_llm.cache.contracts import (
    CacheAllocation,
    GatheredKV,
    KVCacheError,
    KVCacheSpec,
    PagedKV,
    SequenceKVCache,
)
from mini_llm.cache.lengths import SequenceLength
from mini_llm.cache.paged.blocks import BlockAllocator
from mini_llm.cache.paged.store import PagedKVStore
from mini_llm.cache.validation import validate_kv_states
from mini_llm.config import DecoderConfig


class PagedLayerKVCache:
    """One layer of one request, addressing the store through its block table."""

    __slots__ = ("sequence", "layer_index")

    def __init__(self, sequence: "PagedSequenceKVCache", layer_index: int) -> None:
        self.sequence = sequence
        self.layer_index = layer_index

    @property
    def length(self) -> int:
        return self.sequence.length

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Write logical token positions into their physical cache blocks."""

        store = self.sequence.store
        validate_kv_states(keys, values, spec=store.spec, batch_size=1)
        store.write(
            self.layer_index,
            self.sequence.block_table.unsqueeze(0),
            position_ids.view(1, -1),
            keys,
            values,
        )

    def gathered(self) -> GatheredKV:
        return self.sequence.store.gather(
            self.layer_index, self.sequence.block_table, self.length
        )

    def paged(self) -> PagedKV:
        store = self.sequence.store
        block_table = self.sequence.block_table
        return PagedKV(
            keys=store.keys[self.layer_index],
            values=store.values[self.layer_index],
            block_table=block_table.unsqueeze(0),
            max_seqlen_k=block_table.numel() * store.block_size,
        )


class PagedBatchLayerKVCache:
    """One layer of a :class:`PagedBatchKVCache`."""

    __slots__ = ("batch", "layer_index")

    def __init__(self, batch: "PagedBatchKVCache", layer_index: int) -> None:
        self.batch = batch
        self.layer_index = layer_index

    @property
    def length(self) -> int:
        return self.batch.length

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        batch = self.batch
        validate_kv_states(
            keys, values, spec=batch.store.spec, batch_size=len(batch.sequences)
        )
        batch.store.write(
            self.layer_index, batch.block_table, position_ids, keys, values
        )

    def gathered(self) -> GatheredKV:
        """Gather every request's prefix into one zero-padded batch allocation."""

        store = self.batch.store
        views = [
            store.gather(self.layer_index, sequence.block_table, sequence.length)
            for sequence in self.batch.sequences
        ]
        length = max(view.keys.shape[2] for view in views)
        _, heads, _, head_dim = views[0].keys.shape
        batch_shape = (len(views), heads, length, head_dim)
        keys = views[0].keys.new_zeros(batch_shape)
        values = views[0].values.new_zeros(batch_shape)
        for index, view in enumerate(views):
            request_length = view.keys.shape[2]
            keys[index, :, :request_length].copy_(view.keys[0])
            values[index, :, :request_length].copy_(view.values[0])
        return GatheredKV(keys=keys, values=values)

    def paged(self) -> PagedKV:
        store = self.batch.store
        block_table = self.batch.block_table
        return PagedKV(
            keys=store.keys[self.layer_index],
            values=store.values[self.layer_index],
            block_table=block_table,
            max_seqlen_k=block_table.shape[1] * store.block_size,
        )


class PagedBatchKVCache:
    """A batch of requests sharing one paged store and one padded block table.

    Every layer view reads the same table. It is allocated once and rewritten
    in place by :meth:`rebind`, so CUDA graphs capture stable device addresses.
    """

    __slots__ = ("store", "block_table", "sequences", "layers")

    def __init__(
        self,
        store: PagedKVStore,
        sequences: Sequence["PagedSequenceKVCache"],
        *,
        block_table_capacity: int | None = None,
    ) -> None:
        if not sequences:
            raise KVCacheError("a batched cache needs at least one request")
        self.store = store
        block_count = block_table_capacity or max(
            sequence.block_table.numel() for sequence in sequences
        )
        self.block_table = torch.zeros(
            (len(sequences), block_count), dtype=torch.int32, device=store.spec.device
        )
        self.sequences: tuple[PagedSequenceKVCache, ...] = ()
        self.layers = [
            PagedBatchLayerKVCache(self, layer_index)
            for layer_index in range(store.spec.num_layers)
        ]
        self.rebind(sequences)

    def rebind(self, sequences: Sequence["PagedSequenceKVCache"]) -> None:
        """Point the existing block table at another batch of the same shape."""

        if len(sequences) != self.block_table.shape[0]:
            raise KVCacheError(
                f"batch size must stay {self.block_table.shape[0]}, got "
                f"{len(sequences)}"
            )
        for sequence in sequences:
            if sequence.store is not self.store:
                raise KVCacheError("a cache batch must share one paged K/V store")
            if sequence.block_table.numel() > self.block_table.shape[1]:
                raise KVCacheError(
                    f"block table holds {self.block_table.shape[1]} blocks but a "
                    f"request needs {sequence.block_table.numel()}"
                )
        # Entries beyond a request's length are never read, so block zero is
        # safe padding for shorter tables.
        self.block_table.zero_()
        for index, sequence in enumerate(sequences):
            blocks = sequence.block_table
            self.block_table[index, : blocks.numel()].copy_(blocks)
        self.sequences = tuple(sequences)

    @property
    def length(self) -> int:
        return max(sequence.length for sequence in self.sequences)


class PagedSequenceKVCache:
    """Reserved blocks, logical length and layer views for one request."""

    __slots__ = ("store", "capacity", "block_table", "layers", "_lengths", "_released")

    def __init__(
        self, store: PagedKVStore, block_table: torch.Tensor, capacity: int
    ) -> None:
        self.store = store
        self.capacity = capacity
        self.block_table = block_table
        self._lengths = SequenceLength(capacity)
        self._released = False
        self.layers = [
            PagedLayerKVCache(self, layer_index)
            for layer_index in range(store.spec.num_layers)
        ]

    @property
    def spec(self) -> KVCacheSpec:
        return self.store.spec

    def _ensure_active(self) -> None:
        if self._released:
            raise KVCacheError("cache has been released")

    @property
    def length(self) -> int:
        self._ensure_active()
        return self._lengths.length

    @property
    def num_bytes(self) -> int:
        return self.spec.num_bytes(self.block_table.numel() * self.store.block_size)

    def extend(self, token_count: int) -> None:
        self._ensure_active()
        self._lengths.extend(token_count)

    def reset(self) -> None:
        self._ensure_active()
        self._lengths.reset()

    def rollback(self, length: int) -> None:
        self._ensure_active()
        self._lengths.rollback(length)

    def batch_layers(
        self, caches: Sequence[SequenceKVCache]
    ) -> list[PagedBatchLayerKVCache]:
        sequences: list[PagedSequenceKVCache] = []
        for cache in caches:
            if not isinstance(cache, PagedSequenceKVCache) or cache.store is not self.store:
                raise KVCacheError("a cache batch must share one paged K/V store")
            sequences.append(cache)
        return PagedBatchKVCache(self.store, sequences).layers

    def release_blocks(self) -> list[int]:
        """Invalidate this cache and return its physical block IDs."""

        blocks = self.block_table.tolist()
        self.block_table = self.block_table.new_empty(0)
        self._lengths.reset()
        self._released = True
        return blocks


class PagedKVCacheManager:
    """Runtime allocator over one fixed-size block store.

    Requests reserve whole blocks up front and exclusively own them until
    release; sharing and reference counts belong to a later milestone.
    """

    def __init__(
        self,
        config: DecoderConfig,
        max_cache_tokens: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        block_size: int = 16,
    ) -> None:
        if max_cache_tokens <= 0:
            raise KVCacheError(
                f"max_cache_tokens must be positive, got {max_cache_tokens}"
            )
        if block_size <= 0:
            raise KVCacheError(f"block_size must be positive, got {block_size}")
        # Resolve "cuda" to the concrete device caches will report in their spec.
        resolved_device = torch.empty(0, device=device).device
        spec = KVCacheSpec.from_config(config, dtype=dtype, device=resolved_device)
        num_blocks = math.ceil(max_cache_tokens / block_size)
        self.store = PagedKVStore(spec, num_blocks, block_size)
        self.blocks = BlockAllocator(num_blocks)
        self.capacity = num_blocks * block_size
        self.max_sequence_length = config.max_position_embeddings
        self._active: set[PagedSequenceKVCache] = set()
        self._last_allocation: CacheAllocation | None = None

    @property
    def spec(self) -> KVCacheSpec:
        return self.store.spec

    @property
    def block_size(self) -> int:
        return self.store.block_size

    @property
    def num_blocks(self) -> int:
        return self.store.num_blocks

    @property
    def free_blocks(self) -> int:
        return self.blocks.free_blocks

    @property
    def used_blocks(self) -> int:
        return self.blocks.used_blocks

    @property
    def used_tokens(self) -> int:
        return self.used_blocks * self.block_size

    @property
    def active_sequences(self) -> int:
        return len(self._active)

    @property
    def last_allocation(self) -> CacheAllocation | None:
        return self._last_allocation

    def can_allocate(self, capacity: int) -> bool:
        return math.ceil(capacity / self.block_size) <= self.blocks.free_blocks

    def allocate(self, capacity: int) -> PagedSequenceKVCache:
        """Reserve enough exclusive blocks for one logical sequence."""

        if capacity <= 0:
            raise KVCacheError(f"cache capacity must be positive, got {capacity}")
        if capacity > self.max_sequence_length:
            raise KVCacheError(
                f"cache capacity {capacity} exceeds model context length "
                f"{self.max_sequence_length}"
            )
        required_blocks = math.ceil(capacity / self.block_size)
        block_table = torch.tensor(
            self.blocks.allocate(required_blocks),
            dtype=torch.int32,
            device=self.store.spec.device,
        )
        cache = PagedSequenceKVCache(self.store, block_table, capacity)
        self._active.add(cache)
        self._last_allocation = CacheAllocation(cache.num_bytes, capacity)
        return cache

    def release(self, cache: SequenceKVCache) -> None:
        if not isinstance(cache, PagedSequenceKVCache) or cache.store is not self.store:
            raise KVCacheError("cache is not owned by this cache manager")
        if cache not in self._active:
            return
        self._active.remove(cache)
        self.blocks.free(cache.release_blocks())
