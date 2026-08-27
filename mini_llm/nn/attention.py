"""Uncached grouped-query causal self-attention for Qwen3."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.config import Qwen3Config
from mini_llm.nn.norm import RMSNorm, normalize_qwen3_queries_and_keys
from mini_llm.nn.rope import apply_qwen3_rotary_position_embeddings


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


class Qwen3Attention(nn.Module):
    """Qwen3 self-attention without a KV cache.

    The module accepts residual-stream states shaped
    ``[batch, sequence, hidden_size]``.  It projects and reshapes them into
    query and KV heads, applies Q/K RMSNorm and RoPE, shares each KV head among
    its query-head group, performs causal scaled dot-product attention, and
    projects the concatenated query heads back to ``hidden_size``.
    """

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
        if rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {rms_norm_eps}")
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
        self.scaling = 1.0 / math.sqrt(head_dim)
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
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    @classmethod
    def from_config(cls, config: Qwen3Config) -> "Qwen3Attention":
        """Construct attention using a validated Qwen3 configuration."""

        return cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )

    def _split_heads(self, states: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        return states.view(
            batch_size, sequence_length, num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        if not inputs.is_floating_point():
            raise TypeError(
                f"Qwen3Attention requires floating-point input, got {inputs.dtype}"
            )
        if inputs.ndim != 3 or inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                "Qwen3Attention expected input shape [batch, sequence, "
                f"{self.hidden_size}], got {tuple(inputs.shape)}"
            )

        queries = self._split_heads(
            self.q_proj(inputs), self.num_attention_heads
        )
        keys = self._split_heads(self.k_proj(inputs), self.num_key_value_heads)
        values = self._split_heads(self.v_proj(inputs), self.num_key_value_heads)

        queries, keys = normalize_qwen3_queries_and_keys(
            queries,
            keys,
            query_norm=self.q_norm,
            key_norm=self.k_norm,
        )
        queries, keys = apply_qwen3_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        keys = repeat_kv_heads(keys, self.queries_per_kv_head)
        values = repeat_kv_heads(values, self.queries_per_kv_head)
        dropout_p = self.attention_dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=dropout_p,
            is_causal=True,
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
            f"head_dim={self.head_dim}"
        )
