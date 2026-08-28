"""Architecture-neutral decoder and checkpoint-loading mechanics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Iterable, Self

import torch
from torch import nn

from mini_llm.cache import LayerKVCache
from mini_llm.checkpoint import (
    CheckpointValidationError,
    SafeTensorCheckpoint,
    validate_checkpoint,
)
from mini_llm.config import DecoderConfig
from mini_llm.nn import RMSNorm, RotaryEmbedding
from mini_llm.nn.rope import build_position_ids


class DecoderModel(nn.Module):
    """Shared embeddings, positions, layer traversal, and final normalization."""

    def __init__(
        self,
        config: DecoderConfig,
        layers: Iterable[nn.Module],
        *,
        embedding_multiplier: float = 1.0,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not math.isfinite(embedding_multiplier) or embedding_multiplier <= 0:
            raise ValueError(
                "embedding_multiplier must be finite and positive, got "
                f"{embedding_multiplier}"
            )

        self.config = config
        self.embedding_multiplier = embedding_multiplier
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=padding_idx,
        )
        self.layers = nn.ModuleList(layers)
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
                "input_ids must have shape [batch, sequence], got "
                f"{tuple(input_ids.shape)}"
            )
        if input_ids.dtype == torch.bool or input_ids.is_floating_point():
            raise TypeError(
                f"input_ids must use an integer dtype, got {input_ids.dtype}"
            )
        if input_ids.numel() == 0:
            raise ValueError("input_ids must contain at least one token")
        if torch.any(input_ids < 0).item() or torch.any(
            input_ids >= self.config.vocab_size
        ).item():
            raise ValueError(
                "input_ids must be within vocabulary "
                f"[0, {self.config.vocab_size})"
            )

        batch_size, sequence_length = input_ids.shape
        if layer_caches is not None and batch_size != 1:
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
                "position_ids must have the same [batch, sequence] shape as "
                f"input_ids, got {tuple(position_ids.shape)} and "
                f"{tuple(input_ids.shape)}"
            )
        elif layer_caches is not None and not torch.equal(
            position_ids, expected_position_ids
        ):
            raise ValueError(
                "cached position_ids must continue from the cache length "
                f"{position_offset}, got {position_ids.tolist()}"
            )

        hidden_states = self.embed_tokens(input_ids)
        if self.embedding_multiplier != 1.0:
            hidden_states = hidden_states * self.embedding_multiplier
        cosine, sine = self.rotary_emb(
            position_ids, output_dtype=hidden_states.dtype
        )
        for layer_index, layer in enumerate(self.layers):
            layer_cache = (
                None if layer_caches is None else layer_caches[layer_index]
            )
            hidden_states = layer(hidden_states, cosine, sine, cache=layer_cache)
        return self.norm(hidden_states)


class CausalLMBase(nn.Module):
    """Common stateless forward pass and strict Safetensors assignment."""

    config_class: ClassVar[type[DecoderConfig]]
    config: DecoderConfig
    model: DecoderModel

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a stateless, uncached full-sequence forward pass."""

        hidden_states = self.model(input_ids, position_ids=position_ids)
        return self._project_logits(hidden_states)

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def input_device(self) -> torch.device:
        """Device on which token IDs must be created for this model."""

        return self.model.embed_tokens.weight.device

    def materialize_derived_buffers(self, device: torch.device | str) -> None:
        """Rebuild non-checkpoint tensors after model placement."""

        self.model.rotary_emb.materialize(device)

    def load_checkpoint(self, checkpoint: SafeTensorCheckpoint) -> None:
        """Validate and assign learned weights to a meta-constructed model."""

        validate_checkpoint(checkpoint, self.config)
        state_dict = checkpoint.get_tensors()
        try:
            incompatible = self.load_state_dict(state_dict, strict=True, assign=True)
        except RuntimeError as exc:
            raise CheckpointValidationError(
                "checkpoint tensors do not match the "
                f"{type(self).__name__} module hierarchy: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise CheckpointValidationError(
                "strict checkpoint assignment reported missing or unexpected tensors"
            )

        # Checkpoints contain learned parameters but not derived RoPE values.
        self.materialize_derived_buffers("cpu")

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> Self:
        """Build on meta, strictly assign one checkpoint, and return an eval model."""

        config = cls.config_class.from_model_dir(model_dir)
        checkpoint = SafeTensorCheckpoint.from_model_dir(model_dir)
        with torch.device("meta"):
            model = cls(config)
        model.load_checkpoint(checkpoint)
        model.eval()
        return model
