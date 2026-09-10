"""Request-owned block-paged KV cache with a transparent gather backend."""

from __future__ import annotations

import math

import torch

from mini_llm.cache import KVCacheError, LayerKVCacheView, SequenceKVCache
from mini_llm.config import DecoderConfig


class PagedKVCachePool:
    """Global fixed-size block pool shared by independent request caches.

    A physical block ID selects the corresponding K/V block in every decoder
    layer. Handles reserve whole blocks up front and exclusively own them until
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
        if not dtype.is_floating_point:
            raise KVCacheError(f"cache dtype must be floating point, got {dtype}")

        self.block_size = block_size
        self.num_blocks = math.ceil(max_cache_tokens / block_size)
        self.capacity = self.num_blocks * block_size
        self.max_sequence_length = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dtype = dtype
        requested_device = torch.device(device)
        shape = (
            self.num_blocks,
            block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        self.keys = [
            torch.empty(shape, dtype=dtype, device=requested_device)
            for _ in range(self.num_layers)
        ]
        self.values = [
            torch.empty(shape, dtype=dtype, device=requested_device)
            for _ in range(self.num_layers)
        ]
        self.device = self.keys[0].device
        self._active: set[SequenceCacheHandle] = set()
        self._last_allocation_num_bytes = 0
        self._last_allocation_capacity = 0
        # Allocate low IDs first and deterministically reuse released blocks.
        self._free_blocks = list(reversed(range(self.num_blocks)))

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    @property
    def active_sequences(self) -> int:
        return len(self._active)

    @property
    def used_tokens(self) -> int:
        return self.used_blocks * self.block_size

    @property
    def last_allocation_num_bytes(self) -> int:
        return self._last_allocation_num_bytes

    @property
    def last_allocation_capacity(self) -> int:
        return self._last_allocation_capacity

    def allocate(self, capacity: int) -> "SequenceCacheHandle":
        """Reserve enough exclusive blocks for one logical sequence."""

        if capacity <= 0:
            raise KVCacheError(f"cache capacity must be positive, got {capacity}")
        if capacity > self.max_sequence_length:
            raise KVCacheError(
                f"cache capacity {capacity} exceeds model context length "
                f"{self.max_sequence_length}"
            )
        required_blocks = math.ceil(capacity / self.block_size)
        if required_blocks > self.free_blocks:
            raise KVCacheError(
                "KV block pool exhausted: need "
                f"{required_blocks} blocks but only {self.free_blocks} are free"
            )
        block_table = torch.tensor(
            [self._free_blocks.pop() for _ in range(required_blocks)],
            dtype=torch.int32,
            device=self.device,
        )
        handle = SequenceCacheHandle(self, capacity, block_table)
        self._active.add(handle)
        self._last_allocation_num_bytes = handle.num_bytes
        self._last_allocation_capacity = capacity
        return handle

    def release(self, cache: SequenceKVCache) -> None:
        if not isinstance(cache, SequenceCacheHandle) or cache.pool is not self:
            raise KVCacheError("cache is not owned by this paged pool")
        if cache not in self._active:
            return
        self._active.remove(cache)
        blocks = cache._release()
        self._free_blocks.extend(blocks)
        self._free_blocks.sort(reverse=True)

    def _gather(
        self, layer_index: int, handle: "SequenceCacheHandle", length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.keys[layer_index].index_select(0, handle.block_table)
        values = self.values[layer_index].index_select(0, handle.block_table)
        # Page layout is [block, token, kv_head, dim]; SDPA consumes
        # [batch, kv_head, token, dim]. This gather is the reference backend.
        keys = keys.flatten(0, 1)[:length].permute(1, 0, 2).unsqueeze(0)
        values = values.flatten(0, 1)[:length].permute(1, 0, 2).unsqueeze(0)
        return keys, values


class PagedLayerKVCache:
    """One layer's attention-facing view of a request page table."""

    def __init__(self, handle: "SequenceCacheHandle", layer_index: int) -> None:
        self.handle = handle
        self.layer_index = layer_index

    @property
    def length(self) -> int:
        return self.handle._layer_lengths[self.layer_index]

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write K/V and gather the exact logical prefix for reference SDPA."""

        self.write(keys, values, position_ids)
        view = self.view(gathered=True)
        return view.keys, view.values

    def view(self, *, gathered: bool) -> LayerKVCacheView:
        """Return logical contiguous K/V or the physical paged representation."""

        pool = self.handle.pool
        if gathered:
            keys, values = pool._gather(
                self.layer_index, self.handle, self.length
            )
            block_table = None
        else:
            keys = pool.keys[self.layer_index]
            values = pool.values[self.layer_index]
            block_table = self.handle.block_table.unsqueeze(0)
        return LayerKVCacheView(
            keys=keys,
            values=values,
            block_table=block_table,
            capacity=self.handle.capacity,
        )

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Write logical token positions into their physical cache blocks."""

        self.handle._ensure_active()
        pool = self.handle.pool
        if keys.ndim != 4:
            raise KVCacheError(
                "new keys must have shape [1, kv_heads, tokens, head_dim], got "
                f"{tuple(keys.shape)}"
            )
        expected = (1, pool.num_key_value_heads, keys.shape[2], pool.head_dim)
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise KVCacheError(
                f"new K/V must both have shape {expected}, got "
                f"{tuple(keys.shape)} and {tuple(values.shape)}"
            )
        if keys.dtype != pool.dtype or values.dtype != pool.dtype:
            raise KVCacheError(f"new K/V dtype must match cache dtype {pool.dtype}")
        if keys.device != pool.device or values.device != pool.device:
            raise KVCacheError(
                f"new K/V device must match cache device {pool.device}"
            )
        token_count = keys.shape[2]
        if token_count <= 0:
            raise KVCacheError(f"token_count must be positive, got {token_count}")
        start = self.length
        end = start + token_count
        if end > self.handle.capacity:
            raise KVCacheError(
                f"KV cache capacity exceeded: need {end} positions but capacity "
                f"is {self.handle.capacity}"
            )

        source_keys = keys.squeeze(0).transpose(0, 1)
        source_values = values.squeeze(0).transpose(0, 1)
        with torch.no_grad():
            positions = position_ids.flatten()
            table_indices = torch.div(
                positions, pool.block_size, rounding_mode="floor"
            )
            block_offsets = torch.remainder(positions, pool.block_size)
            block_ids = self.handle.block_table.index_select(0, table_indices)
            slots = block_ids * pool.block_size + block_offsets
            pool.keys[self.layer_index].flatten(0, 1).index_copy_(
                0, slots, source_keys
            )
            pool.values[self.layer_index].flatten(0, 1).index_copy_(
                0, slots, source_values
            )
        self.handle._layer_lengths[self.layer_index] = end


class SequenceCacheHandle:
    """Reserved capacity, ordered block table, and lengths for one request."""

    def __init__(
        self,
        pool: PagedKVCachePool,
        capacity: int,
        block_table: torch.Tensor,
    ) -> None:
        self.pool = pool
        self.capacity = capacity
        self.block_table = block_table
        self._layer_lengths = [0] * pool.num_layers
        self._released = False
        self.layers = [
            PagedLayerKVCache(self, layer_index)
            for layer_index in range(pool.num_layers)
        ]

    def _ensure_active(self) -> None:
        if self._released:
            raise KVCacheError("cache handle has been released")

    @property
    def length(self) -> int:
        self._ensure_active()
        lengths = set(self._layer_lengths)
        if len(lengths) != 1:
            raise KVCacheError(
                f"layer cache lengths are inconsistent: {self._layer_lengths}"
            )
        return self._layer_lengths[0]

    @property
    def num_key_value_heads(self) -> int:
        return self.pool.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.pool.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.pool.dtype

    @property
    def device(self) -> torch.device:
        return self.pool.device

    @property
    def num_bytes(self) -> int:
        bytes_per_element = torch.empty((), dtype=self.dtype).element_size()
        return (
            self.block_table.numel()
            * self.pool.block_size
            * self.pool.num_layers
            * self.pool.num_key_value_heads
            * self.pool.head_dim
            * 2
            * bytes_per_element
        )

    def ensure_can_append(self, token_count: int) -> None:
        current_length = self.length
        if token_count <= 0:
            raise KVCacheError(f"token_count must be positive, got {token_count}")
        required = current_length + token_count
        if required > self.capacity:
            raise KVCacheError(
                f"KV cache capacity exceeded: need {required} positions but capacity "
                f"is {self.capacity}"
            )

    def reset(self) -> None:
        self._ensure_active()
        self._layer_lengths[:] = [0] * self.pool.num_layers

    def rollback(self, length: int) -> None:
        self._ensure_active()
        if length < 0 or any(length > current for current in self._layer_lengths):
            raise KVCacheError(f"cannot roll cache back to length {length}")
        self._layer_lengths[:] = [length] * self.pool.num_layers

    def _release(self) -> list[int]:
        """Invalidate this handle and return its physical block IDs."""

        blocks = self.block_table.tolist()
        self.block_table = self.block_table.new_empty(0)
        self._layer_lengths[:] = [0] * self.pool.num_layers
        self._released = True
        return blocks
