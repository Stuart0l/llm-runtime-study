"""Qwen3 decoder model with uncached and dense KV-cache execution."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mini_llm.cache import LayerKVCache, Qwen3KVCache
from mini_llm.checkpoint import (
    CheckpointValidationError,
    SafeTensorCheckpoint,
    validate_qwen3_checkpoint,
)
from mini_llm.config import Qwen3Config
from mini_llm.nn import Qwen3Attention, RMSNorm, RotaryEmbedding, SwiGLUFeedForward
from mini_llm.nn.rope import build_position_ids


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


class Qwen3Model(nn.Module):
    """Token embeddings, decoder stack, and final norm without an LM head."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        layer_caches: list[LayerKVCache] | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Compute hidden states, optionally writing trusted per-layer caches."""
        
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype == torch.bool or input_ids.is_floating_point():
            raise TypeError(f"input_ids must use an integer dtype, got {input_ids.dtype}")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must contain at least one token")
        if torch.any(input_ids < 0).item() or torch.any(
            input_ids >= self.config.vocab_size
        ).item():
            raise ValueError(
                f"input_ids must be within vocabulary [0, {self.config.vocab_size})"
            )

        batch_size, sequence_length = input_ids.shape
        if layer_caches is not None:
            if batch_size != 1:
                raise ValueError("v1 cached execution supports batch size one only")

        expected_position_ids = build_position_ids(
            sequence_length,
            offset=position_offset,
            batch_size=batch_size,
            device=input_ids.device,
        )
        if position_ids is None:
            position_ids = expected_position_ids
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, sequence] shape as input_ids, "
                f"got {tuple(position_ids.shape)} and {tuple(input_ids.shape)}"
            )
        elif layer_caches is not None and not torch.equal(
            position_ids, expected_position_ids
        ):
            raise ValueError(
                "cached position_ids must continue from the cache length "
                f"{position_offset}, got {position_ids.tolist()}"
            )

        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_emb(
            position_ids, output_dtype=hidden_states.dtype
        )
        for layer_index, layer in enumerate(self.layers):
            layer_cache = (
                None if layer_caches is None else layer_caches[layer_index]
            )
            hidden_states = layer(hidden_states, cosine, sine, cache=layer_cache)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 model that returns vocabulary logits for supplied token positions."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._cache: Qwen3KVCache | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a stateless, uncached full-sequence forward pass."""

        hidden_states = self.model(input_ids, position_ids=position_ids)
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
        return self.lm_head(hidden_states)

    @property
    def cache(self) -> Qwen3KVCache | None:
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
        self._cache = Qwen3KVCache(
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

    def load_checkpoint(self, checkpoint: SafeTensorCheckpoint) -> None:
        """Turn a meta-constructed model into an executable CPU model.

        Learned parameters come from Safetensors.  Non-persistent derived
        buffers, which Safetensors intentionally does not contain, are
        materialized in the same operation so no partially loaded model escapes.
        """

        validate_qwen3_checkpoint(checkpoint, self.config)
        state_dict = checkpoint.get_tensors()
        try:
            incompatible = self.load_state_dict(state_dict, strict=True, assign=True)
        except RuntimeError as exc:
            raise CheckpointValidationError(
                f"checkpoint tensors do not match the Qwen3 module hierarchy: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CheckpointValidationError(
                "strict checkpoint assignment reported missing or unexpected tensors"
            )

        # Learned meta parameters have now been replaced by real checkpoint
        # tensors. RoPE is derived and non-persistent, so finish materializing
        # the model by calculating its real inverse-frequency values as well.
        self.model.rotary_emb.materialize("cpu")

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3ForCausalLM":
        """Validate and directly assign a single-file Qwen3 checkpoint on CPU."""

        config = Qwen3Config.from_model_dir(model_dir)
        checkpoint = SafeTensorCheckpoint.from_model_dir(model_dir)

        # Build only the module hierarchy and parameter shapes.  Meta tensors
        # have no backing storage, so this avoids allocating a full set of
        # random FP32 weights before replacing them with the BF16 checkpoint.
        with torch.device("meta"):
            model = cls(config)
        model.load_checkpoint(checkpoint)
        model.eval()
        return model
