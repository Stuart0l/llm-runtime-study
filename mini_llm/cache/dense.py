"""Contiguous per-request KV-cache implementation."""

from __future__ import annotations

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
from mini_llm.cache.validation import validate_kv_states
from mini_llm.config import DecoderConfig

_CAPTURE_BUCKET_SIZE = 16


class DenseLayerKVCache:
    """Key/value storage for one decoder layer.

    Both tensors use layout ``[1, kv_heads, capacity, head_dim]``. The shared
    :class:`SequenceLength` marks the valid prefix; values beyond it are
    allocated but inaccessible.
    """

    __slots__ = ("keys", "values", "spec", "_lengths")

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        lengths: SequenceLength,
        spec: KVCacheSpec,
    ) -> None:
        if keys.ndim != 4 or values.ndim != 4:
            raise KVCacheError(
                "cache tensors must have shape [1, kv_heads, capacity, head_dim]"
            )
        if keys.shape != values.shape:
            raise KVCacheError("key and value cache shapes must match")
        if keys.shape[0] != 1:
            raise KVCacheError("v1 KV cache supports batch size one only")
        if keys.dtype != values.dtype:
            raise KVCacheError("key and value cache dtypes must match")
        if keys.device != values.device:
            raise KVCacheError("key and value caches must use the same device")
        if keys.shape[2] != lengths.capacity:
            raise KVCacheError(
                f"cache tensors hold {keys.shape[2]} positions but capacity is "
                f"{lengths.capacity}"
            )
        self.keys = keys
        self.values = values
        self.spec = spec
        self._lengths = lengths

    @property
    def length(self) -> int:
        return self._lengths.length

    @property
    def capacity(self) -> int:
        return self.keys.shape[2]

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Write logical token positions into contiguous cache storage."""

        validate_kv_states(keys, values, spec=self.spec, batch_size=1)
        # Cache contents are inference state, not part of an autograd graph.
        with torch.no_grad():
            positions = position_ids.flatten()
            self.keys.index_copy_(2, positions, keys)
            self.values.index_copy_(2, positions, values)

    def gathered(self) -> GatheredKV:
        """Return the contiguous prefix used by attention."""

        attention_length = self.length
        if self.keys.is_cuda and torch.cuda.is_current_stream_capturing():
            # Capture must read a fixed-size window, so round the prefix up to
            # a bucket whose masked, zero-initialized tail stays finite.
            attention_length = min(
                self.capacity,
                ((self.length + _CAPTURE_BUCKET_SIZE - 1) // _CAPTURE_BUCKET_SIZE)
                * _CAPTURE_BUCKET_SIZE,
            )
        return GatheredKV(
            keys=self.keys[:, :, :attention_length],
            values=self.values[:, :, :attention_length],
        )

    def paged(self) -> PagedKV | None:
        return None


class DenseBatchLayerKVCache:
    """One decoder layer across a batch of independent contiguous caches."""

    __slots__ = ("layers", "spec")

    def __init__(self, layers: Sequence[DenseLayerKVCache]) -> None:
        if not layers:
            raise KVCacheError("a batched layer cache needs at least one request")
        self.layers = tuple(layers)
        self.spec = self.layers[0].spec

    @property
    def length(self) -> int:
        return max(layer.length for layer in self.layers)

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        validate_kv_states(
            keys, values, spec=self.spec, batch_size=len(self.layers)
        )
        for index, layer in enumerate(self.layers):
            layer.write(
                keys[index : index + 1],
                values[index : index + 1],
                position_ids[index : index + 1],
            )

    def gathered(self) -> GatheredKV:
        """Copy every request's prefix into one zero-padded batch allocation."""

        views = [layer.gathered() for layer in self.layers]
        length = max(view.keys.shape[2] for view in views)
        _, heads, _, head_dim = views[0].keys.shape
        batch_shape = (len(views), heads, length, head_dim)
        keys = views[0].keys.new_zeros(batch_shape)
        values = views[0].values.new_zeros(batch_shape)
        for batch_index, view in enumerate(views):
            request_length = view.keys.shape[2]
            keys[batch_index, :, :request_length].copy_(view.keys[0])
            values[batch_index, :, :request_length].copy_(view.values[0])
        return GatheredKV(keys=keys, values=values)

    def paged(self) -> PagedKV | None:
        return None


class DenseSequenceKVCache:
    """One zero-initialized :class:`DenseLayerKVCache` per decoder layer.

    CUDA graph capture exposes a fixed-size attention bucket beyond the
    logical cache length. Those masked, unwritten slots must remain finite
    because SDPA may read them before applying its attention mask.
    """

    __slots__ = ("spec", "capacity", "layers", "_lengths")

    def __init__(self, spec: KVCacheSpec, capacity: int) -> None:
        self.spec = spec
        self.capacity = capacity
        self._lengths = SequenceLength(capacity)
        shape = (1, spec.num_key_value_heads, capacity, spec.head_dim)
        self.layers = [
            DenseLayerKVCache(
                torch.zeros(shape, dtype=spec.dtype, device=spec.device),
                torch.zeros(shape, dtype=spec.dtype, device=spec.device),
                self._lengths,
                spec,
            )
            for _ in range(spec.num_layers)
        ]

    @property
    def length(self) -> int:
        return self._lengths.length

    @property
    def num_bytes(self) -> int:
        return self.spec.num_bytes(self.capacity)

    def extend(self, token_count: int) -> None:
        self._lengths.extend(token_count)

    def reset(self) -> None:
        self._lengths.reset()

    def rollback(self, length: int) -> None:
        self._lengths.rollback(length)

    def batch_layers(
        self, caches: Sequence[SequenceKVCache]
    ) -> list[DenseBatchLayerKVCache]:
        for cache in caches:
            if not isinstance(cache, DenseSequenceKVCache):
                raise KVCacheError(
                    "a dense cache batch accepts dense caches only, got "
                    f"{type(cache).__name__}"
                )
            if cache.spec != self.spec:
                raise KVCacheError(
                    "a cache batch must share one cache spec, got "
                    f"{cache.spec} and {self.spec}"
                )
        return [
            DenseBatchLayerKVCache([cache.layers[index] for cache in caches])
            for index in range(self.spec.num_layers)
        ]


class DenseKVCacheManager:
    """Runtime allocator for independent contiguous sequence caches."""

    def __init__(
        self,
        config: DecoderConfig,
        max_cache_tokens: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_cache_tokens <= 0:
            raise KVCacheError(
                f"max_cache_tokens must be positive, got {max_cache_tokens}"
            )
        self.config = config
        self.capacity = max_cache_tokens
        # Resolve "cuda" to the concrete device caches will report in their spec.
        resolved_device = torch.empty(0, device=device).device
        self.spec = KVCacheSpec.from_config(
            config, dtype=dtype, device=resolved_device
        )
        self._active: dict[int, DenseSequenceKVCache] = {}
        self._last_allocation: CacheAllocation | None = None

    @property
    def active_sequences(self) -> int:
        return len(self._active)

    @property
    def used_tokens(self) -> int:
        return sum(cache.capacity for cache in self._active.values())

    @property
    def last_allocation(self) -> CacheAllocation | None:
        return self._last_allocation

    def allocate(self, capacity: int) -> DenseSequenceKVCache:
        if capacity <= 0:
            raise KVCacheError(f"cache capacity must be positive, got {capacity}")
        if self.used_tokens + capacity > self.capacity:
            raise KVCacheError(
                "KV cache capacity exhausted: need "
                f"{capacity} tokens but only {self.capacity - self.used_tokens} are free"
            )
        self.config.validate_context_length(capacity)
        cache = DenseSequenceKVCache(self.spec, capacity)
        self._active[id(cache)] = cache
        self._last_allocation = CacheAllocation(cache.num_bytes, capacity)
        return cache

    def release(self, cache: SequenceKVCache) -> None:
        owned = self._active.get(id(cache))
        if owned is None:
            return
        if owned is not cache:
            raise KVCacheError("cache is not owned by this manager")
        del self._active[id(cache)]
