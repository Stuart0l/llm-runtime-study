"""RMS normalization primitives used by Qwen3."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""Normalize each vector by its root-mean-square magnitude.

    For the final dimension of an input vector :math:`x`, this computes

    .. math::

        y = w \odot \frac{x}{\sqrt{\operatorname{mean}(x^2) + \epsilon}}

    Unlike LayerNorm, RMSNorm does not subtract the vector's mean.  Statistics
    are accumulated in FP32 so FP16/BF16 activations do not overflow or lose
    unnecessary precision while squaring and averaging.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_floating_point():
            raise TypeError(f"RMSNorm requires floating-point input, got {inputs.dtype}")
        if inputs.ndim == 0 or inputs.shape[-1] != self.hidden_size:
            actual = None if inputs.ndim == 0 else inputs.shape[-1]
            raise ValueError(
                f"RMSNorm expected final dimension {self.hidden_size}, got {actual}"
            )

        input_dtype = inputs.dtype
        working = inputs.to(torch.float32)
        mean_square = working.square().mean(dim=-1, keepdim=True)
        normalized = working * torch.rsqrt(mean_square + self.eps)
        return self.weight * normalized.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


def normalize_qwen3_queries_and_keys(
    queries: torch.Tensor,
    keys: torch.Tensor,
    *,
    query_norm: RMSNorm,
    key_norm: RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply independent RMSNorm weights to each Qwen3 query and key head.

    Expected layouts are ``[batch, heads, sequence, head_dim]``.  Query and
    key head counts may differ because Qwen3 uses grouped-query attention.
    """

    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError(
            "Qwen3 Q/K normalization expects rank-4 tensors shaped "
            "[batch, heads, sequence, head_dim]"
        )
    if queries.shape[0] != keys.shape[0]:
        raise ValueError(
            "queries and keys must have the same batch size, got "
            f"{queries.shape[0]} and {keys.shape[0]}"
        )
    if queries.shape[2] != keys.shape[2]:
        raise ValueError(
            "queries and keys must have the same sequence length, got "
            f"{queries.shape[2]} and {keys.shape[2]}"
        )
    if queries.shape[-1] != query_norm.hidden_size:
        raise ValueError(
            "query head dimension does not match query_norm: "
            f"{queries.shape[-1]} != {query_norm.hidden_size}"
        )
    if keys.shape[-1] != key_norm.hidden_size:
        raise ValueError(
            "key head dimension does not match key_norm: "
            f"{keys.shape[-1]} != {key_norm.hidden_size}"
        )
    return query_norm(queries), key_norm(keys)
