"""Shape, dtype and device checks for K/V states entering a cache."""

from __future__ import annotations

import torch

from mini_llm.cache.contracts import KVCacheError, KVCacheSpec


def validate_kv_states(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    spec: KVCacheSpec,
    batch_size: int,
) -> int:
    """Check new K/V against ``spec`` and return their token count."""

    if keys.ndim != 4:
        raise KVCacheError(
            "new keys must have shape [batch, kv_heads, tokens, head_dim], got "
            f"{tuple(keys.shape)}"
        )
    expected = (batch_size, spec.num_key_value_heads, keys.shape[2], spec.head_dim)
    if tuple(keys.shape) != expected or tuple(values.shape) != expected:
        raise KVCacheError(
            f"new K/V must both have shape {expected}, got "
            f"{tuple(keys.shape)} and {tuple(values.shape)}"
        )
    if keys.dtype != spec.dtype or values.dtype != spec.dtype:
        raise KVCacheError(f"new K/V dtype must match cache dtype {spec.dtype}")
    if keys.device != spec.device or values.device != spec.device:
        raise KVCacheError(f"new K/V device must match cache device {spec.device}")
    token_count = keys.shape[2]
    if token_count <= 0:
        raise KVCacheError(f"token_count must be positive, got {token_count}")
    return token_count
