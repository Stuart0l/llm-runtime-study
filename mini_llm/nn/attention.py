"""Grouped-query causal self-attention with optional cached K/V states."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.cache import LayerKVCache
from mini_llm.nn.norm import RMSNorm, normalize_qwen3_queries_and_keys
from mini_llm.nn.rope import apply_rotary_position_embeddings


def repeat_kv_heads(states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat each KV head for the query heads that share it.

    ``states`` must use layout ``[batch, kv_heads, sequence, head_dim]``.
    Qwen3-0.6B has two query heads per KV head, so its ``repeats`` value is 2.
    """

    if states.ndim != 4:
        raise ValueError(
            "KV states must have shape [batch, kv_heads, sequence, head_dim]"
        )
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if repeats == 1:
        return states
    return states.repeat_interleave(repeats, dim=1)


class GroupedQueryAttention(nn.Module):
    """Shared GQA projection, RoPE, masking, cache, and SDPA mechanics.

    The module accepts residual-stream states shaped
    ``[batch, sequence, hidden_size]``.  It projects and reshapes them into
    query and KV heads, applies an architecture hook followed by RoPE, shares
    each KV head among its query-head group, performs causal scaled dot-product
    attention, and projects the result back to ``hidden_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        attention_scale: float | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        positive_dimensions = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
        }
        for name, value in positive_dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, got "
                f"{num_attention_heads} and {num_key_value_heads}"
            )
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if attention_scale is not None and (
            not math.isfinite(attention_scale) or attention_scale <= 0
        ):
            raise ValueError(
                f"attention_scale must be finite and positive, got {attention_scale}"
            )
        if not isinstance(attention_bias, bool):
            raise TypeError(
                f"attention_bias must be boolean, got {type(attention_bias).__name__}"
            )
        if attention_dropout < 0 or attention_dropout >= 1:
            raise ValueError("attention_dropout must satisfy 0 <= value < 1")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.query_projection_size = num_attention_heads * head_dim
        self.kv_projection_size = num_key_value_heads * head_dim
        self.queries_per_kv_head = num_attention_heads // num_key_value_heads
        self.scaling = (
            1.0 / math.sqrt(head_dim)
            if attention_scale is None
            else attention_scale
        )
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            hidden_size, self.query_projection_size, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, self.kv_projection_size, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, self.kv_projection_size, bias=attention_bias
        )
        self.o_proj = nn.Linear(self.query_projection_size, hidden_size, bias=False)

    def _split_heads(self, states: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        return states.view(
            batch_size, sequence_length, num_heads, self.head_dim
        ).transpose(1, 2)

    def _normalize_queries_and_keys(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Architecture hook; Granite uses the projected values unchanged."""

        return queries, keys

    def forward(
        self,
        inputs: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        *,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        if not inputs.is_floating_point():
            raise TypeError(
                f"{type(self).__name__} requires floating-point input, "
                f"got {inputs.dtype}"
            )
        if inputs.ndim != 3 or inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected input shape [batch, sequence, "
                f"{self.hidden_size}], got {tuple(inputs.shape)}"
            )

        queries = self._split_heads(
            self.q_proj(inputs), self.num_attention_heads
        )
        keys = self._split_heads(self.k_proj(inputs), self.num_key_value_heads)
        values = self._split_heads(self.v_proj(inputs), self.num_key_value_heads)

        queries, keys = self._normalize_queries_and_keys(queries, keys)
        queries, keys = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        attention_mask = None
        is_causal = True
        if cache is not None:
            past_length = cache.length
            keys, values = cache.append(keys, values)
            if past_length > 0 and inputs.shape[1] == 1:
                # A single decode query is at the final absolute position, so
                # every valid cached key is in its past and is visible.
                is_causal = False
            elif past_length > 0:
                # For chunked appends, ordinary is_causal=True would align its
                # triangular mask to the upper-left and hide most cached keys.
                # Build an absolute-position mask with key_position <= query_position.
                query_positions = past_length + torch.arange(
                    inputs.shape[1], device=inputs.device
                )
                key_positions = torch.arange(keys.shape[2], device=inputs.device)
                attention_mask = key_positions.unsqueeze(
                    0
                ) <= query_positions.unsqueeze(1)
                is_causal = False

        keys = repeat_kv_heads(keys, self.queries_per_kv_head)
        values = repeat_kv_heads(values, self.queries_per_kv_head)
        dropout_p = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=self.scaling,
        )

        batch_size, _, sequence_length, _ = attended.shape
        concatenated = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.query_projection_size
        )
        return self.o_proj(concatenated)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, scaling={self.scaling}"
        )


class Qwen3Attention(GroupedQueryAttention):
    """Grouped-query attention with Qwen3's learned per-head Q/K norms."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        if rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {rms_norm_eps}")
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
        )
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def _normalize_queries_and_keys(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize_qwen3_queries_and_keys(
            queries,
            keys,
            query_norm=self.q_norm,
            key_norm=self.k_norm,
        )


class GraniteAttention(GroupedQueryAttention):
    """Granite GQA without post-projection Q/K normalization."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        attention_scale: float,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        # Granite was trained with maximal-update parameterization (muP), whose
        # width-scaling rules include an explicit attention multiplier. For
        # this model, head_dim=64 and attention_scale=1/64=0.015625. Pass that
        # value directly to SDPA: applying its usual 1/sqrt(head_dim) scaling
        # as well would make inference differ from the model's training rule.
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attention_scale=attention_scale,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
        )
