"""Public KV-cache contracts and backend selection."""

from mini_llm.cache.contracts import (
    CacheAllocation,
    GatheredKV,
    KVCacheError,
    KVCacheManager,
    KVCacheSpec,
    LayerKVCache,
    PagedKV,
    SequenceKVCache,
)
from mini_llm.cache.factory import CacheBackend, create_kv_cache_manager

__all__ = [
    "CacheAllocation",
    "CacheBackend",
    "GatheredKV",
    "KVCacheError",
    "KVCacheManager",
    "KVCacheSpec",
    "LayerKVCache",
    "PagedKV",
    "SequenceKVCache",
    "create_kv_cache_manager",
]
