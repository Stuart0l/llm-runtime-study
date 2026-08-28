"""Qwen3-specific decoder layer and causal language model."""

from __future__ import annotations

import torch
from torch import nn

from mini_llm.cache import LayerKVCache
from mini_llm.config import Qwen3Config
from mini_llm.modeling import CausalLMBase, DecoderModel
from mini_llm.nn import Qwen3Attention, RMSNorm, SwiGLUFeedForward


class Qwen3DecoderLayer(nn.Module):
    """One pre-normalized Qwen3 attention and MLP block."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )
        self.mlp = SwiGLUFeedForward(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

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
        hidden_states = attention_residual + hidden_states

        mlp_residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return mlp_residual + hidden_states


class Qwen3Model(DecoderModel):
    """Token embeddings, decoder stack, and final norm without an LM head."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(
            config,
            (Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)),
        )


class Qwen3ForCausalLM(CausalLMBase):
    """Qwen3 model that returns vocabulary logits for supplied token positions."""

    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
