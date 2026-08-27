"""Uncached Qwen3 decoder model assembled from the local primitives."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

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
        self.self_attn = Qwen3Attention.from_config(config)
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
    ) -> torch.Tensor:
        attention_residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cosine, sine)
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
    ) -> torch.Tensor:
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
        if position_ids is None:
            position_ids = build_position_ids(
                sequence_length,
                batch_size=batch_size,
                device=input_ids.device,
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same [batch, sequence] shape as input_ids, "
                f"got {tuple(position_ids.shape)} and {tuple(input_ids.shape)}"
            )

        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_emb(
            position_ids, output_dtype=hidden_states.dtype
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cosine, sine)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Uncached Qwen3 model that returns vocabulary logits for every token."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)

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
