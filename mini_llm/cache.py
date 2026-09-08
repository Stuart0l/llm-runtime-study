"""Backend-neutral KV-cache contracts shared by models and runtimes."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch


class KVCacheError(ValueError):
    """Raised when cache tensors or logical positions are invalid."""


class LayerKVCache(Protocol):
    """Storage-independent interface consumed by decoder attention."""

    @property
    def length(self) -> int: ...

    def append(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class SequenceKVCache(Protocol):
    """Backend-neutral sequence cache consumed by a causal language model."""

    layers: Sequence[LayerKVCache]
    capacity: int
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    @property
    def length(self) -> int: ...

    @property
    def num_bytes(self) -> int: ...

    def ensure_can_append(self, token_count: int) -> None: ...
    def reset(self) -> None: ...
    def rollback(self, length: int) -> None: ...


class KVCacheManager(Protocol):
    """Runtime-owned allocator and releaser for sequence caches."""

    capacity: int

    @property
    def active_sequences(self) -> int: ...

    @property
    def used_tokens(self) -> int: ...

    @property
    def last_allocation_num_bytes(self) -> int: ...

    @property
    def last_allocation_capacity(self) -> int: ...

    def allocate(self, capacity: int) -> SequenceKVCache: ...
    def release(self, cache: SequenceKVCache) -> None: ...
