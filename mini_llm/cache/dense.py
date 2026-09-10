"""Contiguous per-request KV-cache implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_llm.cache import KVCacheError, SequenceKVCache
from mini_llm.config import DecoderConfig

_CAPTURE_BUCKET_SIZE = 16


@dataclass(slots=True)
class DenseLayerKVCache:
    """Key/value storage for one decoder layer.

    Both tensors use layout ``[1, kv_heads, capacity, head_dim]``.  ``length``
    marks the valid prefix; values beyond it are allocated but inaccessible.
    """

    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0

    def __post_init__(self) -> None:
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise KVCacheError(
                "cache tensors must have shape [1, kv_heads, capacity, head_dim]"
            )
        if self.keys.shape != self.values.shape:
            raise KVCacheError("key and value cache shapes must match")
        if self.keys.shape[0] != 1:
            raise KVCacheError("v1 KV cache supports batch size one only")
        if self.keys.dtype != self.values.dtype:
            raise KVCacheError("key and value cache dtypes must match")
        if self.keys.device != self.values.device:
            raise KVCacheError("key and value caches must use the same device")
        if not 0 <= self.length <= self.capacity:
            raise KVCacheError(
                f"cache length must be within [0, {self.capacity}], got {self.length}"
            )

    @property
    def capacity(self) -> int:
        return self.keys.shape[2]

    @property
    def num_key_value_heads(self) -> int:
        return self.keys.shape[1]

    @property
    def head_dim(self) -> int:
        return self.keys.shape[3]

    def ensure_can_append(self, token_count: int) -> None:
        if token_count <= 0:
            raise KVCacheError(f"token_count must be positive, got {token_count}")
        required = self.length + token_count
        if required > self.capacity:
            raise KVCacheError(
                f"KV cache capacity exceeded: need {required} positions but capacity "
                f"is {self.capacity}"
            )

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new positions and return views of the complete valid prefix."""

        if keys.ndim != 4:
            raise KVCacheError(
                "new keys must have shape [1, kv_heads, tokens, head_dim], got "
                f"{tuple(keys.shape)}"
            )
        expected_shape = (1, self.num_key_value_heads, keys.shape[2], self.head_dim)
        if tuple(keys.shape) != expected_shape:
            raise KVCacheError(
                "new keys must have shape [1, kv_heads, tokens, head_dim] = "
                f"{expected_shape}, got {tuple(keys.shape)}"
            )
        if values.shape != keys.shape:
            raise KVCacheError(
                f"new values must match key shape {tuple(keys.shape)}, got "
                f"{tuple(values.shape)}"
            )
        if keys.dtype != self.keys.dtype or values.dtype != self.values.dtype:
            raise KVCacheError(
                f"new K/V dtype must match cache dtype {self.keys.dtype}"
            )
        if keys.device != self.keys.device or values.device != self.values.device:
            raise KVCacheError(
                f"new K/V device must match cache device {self.keys.device}"
            )
        token_count = keys.shape[2]
        self.ensure_can_append(token_count)
        start = self.length
        end = start + token_count
        # Cache contents are inference state, not part of an autograd graph.
        with torch.no_grad():
            positions = position_ids.flatten()
            self.keys.index_copy_(2, positions, keys)
            self.values.index_copy_(2, positions, values)
        self.length = end
        attention_length = end
        if keys.is_cuda and torch.cuda.is_current_stream_capturing():
            attention_length = min(
                self.capacity,
                ((end + _CAPTURE_BUCKET_SIZE - 1) // _CAPTURE_BUCKET_SIZE)
                * _CAPTURE_BUCKET_SIZE,
            )
        return (
            self.keys[:, :, :attention_length],
            self.values[:, :, :attention_length],
        )

    def reset(self) -> None:
        """Logically empty the cache without reallocating or clearing storage."""

        self.length = 0


class DenseKVCache:
    """One preallocated :class:`DenseLayerKVCache` for every decoder layer."""

    def __init__(
        self,
        config: DecoderConfig,
        capacity: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        config.validate_context_length(capacity)
        if not dtype.is_floating_point:
            raise KVCacheError(f"cache dtype must be floating point, got {dtype}")

        self.capacity = capacity
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (1, self.num_key_value_heads, capacity, self.head_dim)
        self.layers = [
            DenseLayerKVCache(
                keys=torch.empty(shape, dtype=dtype, device=self.device),
                values=torch.empty(shape, dtype=dtype, device=self.device),
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.device = self.layers[0].keys.device

    @property
    def length(self) -> int:
        lengths = {layer.length for layer in self.layers}
        if len(lengths) != 1:
            raise KVCacheError(
                f"layer cache lengths are inconsistent: {sorted(lengths)}"
            )
        return next(iter(lengths))

    @property
    def num_bytes(self) -> int:
        return sum(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
            for layer in self.layers
        )

    def ensure_can_append(self, token_count: int) -> None:
        current_length = self.length
        required = current_length + token_count
        if token_count <= 0:
            raise KVCacheError(f"token_count must be positive, got {token_count}")
        if required > self.capacity:
            raise KVCacheError(
                f"KV cache capacity exceeded: need {required} positions but capacity "
                f"is {self.capacity}"
            )

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def rollback(self, length: int) -> None:
        if length < 0 or any(length > layer.length for layer in self.layers):
            raise KVCacheError(f"cannot roll cache back to length {length}")
        for layer in self.layers:
            layer.length = length


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
        self.dtype = dtype
        self.device = torch.device(device)
        self._active: dict[int, DenseKVCache] = {}
        self._last_allocation_num_bytes = 0
        self._last_allocation_capacity = 0

    @property
    def active_sequences(self) -> int:
        return len(self._active)

    @property
    def used_tokens(self) -> int:
        return sum(cache.capacity for cache in self._active.values())

    @property
    def last_allocation_num_bytes(self) -> int:
        return self._last_allocation_num_bytes

    @property
    def last_allocation_capacity(self) -> int:
        return self._last_allocation_capacity

    def allocate(self, capacity: int) -> DenseKVCache:
        if capacity <= 0:
            raise KVCacheError(f"cache capacity must be positive, got {capacity}")
        if self.used_tokens + capacity > self.capacity:
            raise KVCacheError(
                "KV cache capacity exhausted: need "
                f"{capacity} tokens but only {self.capacity - self.used_tokens} are free"
            )
        cache = DenseKVCache(
            self.config, capacity, dtype=self.dtype, device=self.device
        )
        self._active[id(cache)] = cache
        self._last_allocation_num_bytes = cache.num_bytes
        self._last_allocation_capacity = capacity
        return cache

    def release(self, cache: SequenceKVCache) -> None:
        owned = self._active.get(id(cache))
        if owned is None:
            return
        if owned is not cache:
            raise KVCacheError("cache is not owned by this manager")
        del self._active[id(cache)]
