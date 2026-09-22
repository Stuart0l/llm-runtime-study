"""Backend-neutral KV-cache contracts shared by models and runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from mini_llm.config import DecoderConfig


class KVCacheError(ValueError):
    """Raised when cache tensors or logical positions are invalid."""


@dataclass(frozen=True, slots=True)
class KVCacheSpec:
    """Shape, dtype and placement every cache tensor of a model must have."""

    num_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_config(
        cls,
        config: DecoderConfig,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> KVCacheSpec:
        if not dtype.is_floating_point:
            raise KVCacheError(f"cache dtype must be floating point, got {dtype}")
        return cls(
            num_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=torch.device(device),
        )

    def num_bytes(self, tokens: int) -> int:
        """Bytes needed to hold ``tokens`` positions of K and V for every layer."""

        per_token = self.num_layers * 2 * self.num_key_value_heads * self.head_dim
        return per_token * tokens * torch.empty(0, dtype=self.dtype).element_size()


@dataclass(frozen=True, slots=True)
class GatheredKV:
    """Contiguous K/V prefix laid out for scaled-dot-product attention."""

    keys: torch.Tensor  # [batch, kv_heads, length, head_dim]
    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class PagedKV:
    """Whole block-paged store for one layer plus the block table addressing it."""

    keys: torch.Tensor  # [blocks, block_size, kv_heads, head_dim]
    values: torch.Tensor
    block_table: torch.Tensor  # [batch, blocks] int32
    max_seqlen_k: int


@dataclass(frozen=True, slots=True)
class CacheAllocation:
    """Size of the most recent allocation served by a manager."""

    num_bytes: int
    capacity: int


@dataclass(frozen=True, slots=True)
class LayerKVCacheView:
    """K/V tensors plus metadata describing their attention layout."""

    keys: torch.Tensor
    values: torch.Tensor
    block_table: torch.Tensor | None
    capacity: int



class LayerKVCache(Protocol):
    """Storage-independent interface consumed by decoder attention."""

    @property
    def length(self) -> int: ...

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Store K/V at ``position_ids``; never changes the logical length."""
        ...

    def gathered(self) -> GatheredKV: ...

    def paged(self) -> PagedKV | None:
        """Block-paged view, or ``None`` when the storage is not block-paged."""
        ...

    def view(self, *, gathered: bool) -> LayerKVCacheView: ...

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class SequenceKVCache(Protocol):
    """Backend-neutral sequence cache consumed by a causal language model."""

    spec: KVCacheSpec
    layers: Sequence[LayerKVCache]
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    capacity: int

    @property
    def length(self) -> int: ...

    @property
    def num_bytes(self) -> int: ...

    def extend(self, token_count: int) -> None:
        """Validate and commit ``token_count`` new positions for every layer.

        The runtime calls this before the forward pass, so all layers observe
        the final length and replayed device writes need no length bookkeeping.
        A caller that extends and then fails must ``rollback``.
        """
        ...

    def reset(self) -> None: ...
    def rollback(self, length: int) -> None: ...

    def ensure_can_append(self, token_count: int) -> None: ...
    def advance(self, token_count: int) -> None: ...

    def batch_layers(
        self, caches: Sequence["SequenceKVCache"]
    ) -> list[LayerKVCache]:
        """One batched layer view per layer over ``caches``, which must include self.

        Raises ``KVCacheError`` if a member belongs to another backend or store.
        """
        ...


class KVCacheManager(Protocol):
    """Runtime-owned allocator and releaser for sequence caches."""

    capacity: int

    @property
    def active_sequences(self) -> int: ...

    @property
    def used_tokens(self) -> int: ...

    @property
    def last_allocation(self) -> CacheAllocation | None: ...

    @property
    def last_allocation_num_bytes(self) -> int: ...

    @property
    def last_allocation_capacity(self) -> int: ...

    def allocate(self, capacity: int) -> SequenceKVCache: ...
    def release(self, cache: SequenceKVCache) -> None: ...
