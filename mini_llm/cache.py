"""Preallocated dense KV cache shared by decoder-only architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_llm.config import DecoderConfig


class KVCacheError(ValueError):
    """Raised when cache tensors or logical positions are invalid."""


@dataclass(slots=True)
class LayerKVCache:
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
        self, keys: torch.Tensor, values: torch.Tensor
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
            self.keys[:, :, start:end].copy_(keys)
            self.values[:, :, start:end].copy_(values)
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def reset(self) -> None:
        """Logically empty the cache without reallocating or clearing storage."""

        self.length = 0


class DenseKVCache:
    """One preallocated :class:`LayerKVCache` for every decoder layer."""

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
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (1, self.num_key_value_heads, capacity, self.head_dim)
        self.layers = [
            LayerKVCache(
                keys=torch.empty(shape, dtype=dtype, device=self.device),
                values=torch.empty(shape, dtype=dtype, device=self.device),
            )
            for _ in range(self.num_layers)
        ]

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
