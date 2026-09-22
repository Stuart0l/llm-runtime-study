"""Backend selection for runtime-owned KV cache managers."""

from __future__ import annotations

from typing import Literal

import torch

from mini_llm.cache.contracts import KVCacheError, KVCacheManager
from mini_llm.cache.dense import DenseKVCacheManager
from mini_llm.cache.paged import PagedKVCacheManager
from mini_llm.config import DecoderConfig

CacheBackend = Literal["paged", "dense"]


def create_kv_cache_manager(
    backend: CacheBackend,
    config: DecoderConfig,
    max_cache_tokens: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> KVCacheManager:
    """Build the cache manager for ``backend``."""

    if backend == "paged":
        return PagedKVCacheManager(
            config, max_cache_tokens, dtype=dtype, device=device
        )
    if backend == "dense":
        return DenseKVCacheManager(
            config, max_cache_tokens, dtype=dtype, device=device
        )
    raise KVCacheError(
        f"unsupported cache backend {backend!r}; expected 'paged' or 'dense'"
    )
