"""Granite 3.1 sparse-MoE decoder and tied-output language model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mini_llm.cache import LayerKVCache
from mini_llm.config import GraniteMoeConfig
from mini_llm.modeling import CausalLMBase, DecoderModel
from mini_llm.nn import GraniteAttention, GraniteMoeBlock, RMSNorm


class GraniteMoeDecoderLayer(nn.Module):
    """One pre-normalized Granite attention and sparse-MoE block."""

    def __init__(self, config: GraniteMoeConfig) -> None:
        super().__init__()
        self.self_attn = GraniteAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_scale=config.attention_multiplier,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )
        self.block_sparse_moe = GraniteMoeBlock(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.residual_multiplier = config.residual_multiplier

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        *,
        cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        attention_residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cosine, sine, cache=cache)
        hidden_states = (
            attention_residual + hidden_states * self.residual_multiplier
        )

        moe_residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.block_sparse_moe(hidden_states)
        return moe_residual + hidden_states * self.residual_multiplier


class GraniteMoeModel(DecoderModel):
    """Granite embeddings, configured sparse decoder layers, and final RMSNorm."""

    def __init__(self, config: GraniteMoeConfig) -> None:
        super().__init__(
            config,
            (
                GraniteMoeDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ),
            embedding_multiplier=config.embedding_multiplier,
            padding_idx=config.pad_token_id,
        )


class GraniteMoeForCausalLM(CausalLMBase):
    """Granite decoder with an embedding-tied, scaled vocabulary projection."""

    config_class = GraniteMoeConfig

    def __init__(self, config: GraniteMoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GraniteMoeModel(config)

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Granite ties its output projection to the token embedding matrix. The
        # checkpoint therefore has no independent lm_head.weight tensor.
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        return logits / self.config.logits_scaling
