"""Qwen3-specific decoder layer and causal language model."""

from __future__ import annotations

import torch
from torch import nn

from mini_llm.cache import DenseKVCache, LayerKVCache
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
        self._cache: DenseKVCache | None = None

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reset the owned cache, write a complete prompt, and return its logits."""

        if self._cache is None:
            raise RuntimeError("call setup_cache(capacity) before prefill")
        self._cache.reset()
        return self._cached_forward(input_ids)

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Append exactly one token to the owned cache and return its logits."""

        if self._cache is None:
            raise RuntimeError("call setup_cache(capacity) before decode")
        if self._cache.length == 0:
            raise RuntimeError("call prefill(input_ids) before decode")
        if input_ids.ndim != 2 or input_ids.shape != (1, 1):
            raise ValueError(
                "decode input_ids must have shape [1, 1], got "
                f"{tuple(input_ids.shape)}"
            )
        return self._cached_forward(input_ids)

    def _cached_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run with the cache whose stable properties setup_cache established."""

        assert self._cache is not None
        sequence_length = input_ids.shape[1]
        self._cache.ensure_can_append(sequence_length)
        past_length = self._cache.length
        hidden_states = self.model(
            input_ids,
            layer_caches=self._cache.layers,
            position_offset=past_length,
        )
        expected_length = past_length + sequence_length
        if self._cache.length != expected_length:
            raise RuntimeError(
                f"KV cache length should be {expected_length}, "
                f"got {self._cache.length}"
            )
        return self._project_logits(hidden_states)

    @property
    def cache(self) -> DenseKVCache | None:
        """The model-owned cache for its one active request, if allocated."""

        return self._cache

    def setup_cache(self, capacity: int) -> None:
        """Ensure reusable model-owned cache storage for one active request."""

        parameter = self.model.embed_tokens.weight
        if parameter.is_meta:
            raise RuntimeError("cannot create a KV cache for an unmaterialized model")

        # Interactive requests commonly need similar capacities. Keep the
        # existing allocation when it is large enough; prefill will reset it
        # again before writing the next prompt. If it is too small, release it
        # before constructing the replacement so MPS never briefly holds both
        # dense caches in unified memory.
        if self._cache is not None and self._cache.capacity >= capacity:
            self._cache.reset()
            return
        self._cache = None
        self._cache = DenseKVCache(
            self.config,
            capacity,
            dtype=parameter.dtype,
            device=parameter.device,
        )

    def _apply(self, fn, recurse: bool = True):
        # Device or dtype transformations invalidate the compatibility that
        # setup_cache established once. Reallocate afterward instead of paying
        # to validate stable cache properties during every generated token.
        self._cache = None
        return super()._apply(fn, recurse=recurse)
