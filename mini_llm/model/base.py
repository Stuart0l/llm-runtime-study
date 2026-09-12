"""Architecture-neutral decoder and checkpoint-loading mechanics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Iterable, Self, Sequence

import torch
from torch import nn

from mini_llm.cache import LayerKVCache, SequenceKVCache
from mini_llm.cache.batch import PaddedBatchLayerKVCache
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
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        layer_caches: Sequence[LayerKVCache] | None = None,
    ) -> torch.Tensor:
        """Compute hidden states from tensor inputs and optional KV caches."""

        batch_size, sequence_length = input_ids.shape
        if position_ids is None:
            position_ids = build_position_ids(
                sequence_length,
                batch_size=batch_size,
                device=input_ids.device,
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
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                position_ids=position_ids,
                cache=layer_cache,
            )
        return self.norm(hidden_states)


class CausalLMBase(nn.Module):
    """Common stateless, cached, and strict checkpoint-loading mechanics."""

    config_class: ClassVar[type[DecoderConfig]]
    config: DecoderConfig
    model: DecoderModel

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Validate and run a stateless, uncached request."""

        self._validate_input_ids(input_ids)
        if position_ids is not None:
            if position_ids.shape != input_ids.shape:
                raise ValueError(
                    "position_ids must have the same [batch, sequence] shape as "
                    f"input_ids, got {tuple(position_ids.shape)} and "
                    f"{tuple(input_ids.shape)}"
                )
            if position_ids.dtype == torch.bool or position_ids.is_floating_point():
                raise TypeError(
                    f"position_ids must use an integer dtype, got {position_ids.dtype}"
                )
            if torch.any(position_ids < 0).item():
                raise ValueError("position_ids must be non-negative")
            if torch.any(
                position_ids >= self.config.max_position_embeddings
            ).item():
                maximum = int(position_ids.max().item())
                raise ValueError(
                    f"position ID {maximum} exceeds the model limit "
                    f"{self.config.max_position_embeddings - 1}"
                )

        hidden_states = self.model(input_ids, position_ids=position_ids)
        return self._project_logits(hidden_states)

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        """Validate token tensors at each public request boundary."""

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

    def _project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        cache: SequenceKVCache,
    ) -> torch.Tensor:
        """Reset one explicit request cache, write its prompt, and return logits."""

        self._validate_cache(cache)
        cache.reset()
        return self._cached_forward(input_ids, cache)

    def decode(
        self,
        input_ids: torch.Tensor,
        *,
        caches: Sequence[SequenceKVCache],
    ) -> torch.Tensor:
        """Decode one token per request, including a one-request batch."""

        self._validate_input_ids(input_ids)
        if input_ids.shape[1] != 1:
            raise ValueError(
                "decode input_ids must have shape [batch, 1], got "
                f"{tuple(input_ids.shape)}"
            )
        if input_ids.shape[0] != len(caches):
            raise ValueError(
                f"input batch size {input_ids.shape[0]} does not match "
                f"{len(caches)} caches"
            )

        previous_lengths = []
        for cache in caches:
            self._validate_cache(cache)
            if cache.length == 0:
                raise RuntimeError("prefill every cache before batched decode")
            cache.ensure_can_append(1)
            previous_lengths.append(cache.length)

        position_ids = torch.tensor(
            previous_lengths, dtype=torch.long, device=input_ids.device
        ).unsqueeze(1)
        if len(caches) == 1:
            # Preserve the concrete paged layer so CUDA attention can select
            # its direct varlen kernel instead of the gathered SDPA fallback.
            layer_caches = caches[0].layers
        else:
            layer_caches = [
                PaddedBatchLayerKVCache(
                    [cache.layers[layer_index] for cache in caches]
                )
                for layer_index in range(self.config.num_hidden_layers)
            ]
        try:
            hidden_states = self.model(
                input_ids,
                position_ids=position_ids,
                layer_caches=layer_caches,
            )
            for cache, previous_length in zip(caches, previous_lengths):
                expected_length = previous_length + 1
                if cache.length != expected_length:
                    raise RuntimeError(
                        f"KV cache length should be {expected_length}, "
                        f"got {cache.length}"
                    )
            return self._project_logits(hidden_states)
        except Exception:
            for cache, previous_length in zip(caches, previous_lengths):
                cache.rollback(previous_length)
            raise

    def _cached_forward(
        self, input_ids: torch.Tensor, cache: SequenceKVCache
    ) -> torch.Tensor:
        """Run one cached forward pass through a backend-neutral cache view."""

        self._validate_input_ids(input_ids)
        if input_ids.shape[0] != 1:
            raise ValueError("v1 cached execution supports batch size one only")
        sequence_length = input_ids.shape[1]
        cache.ensure_can_append(sequence_length)
        past_length = cache.length
        position_ids = build_position_ids(
            sequence_length,
            offset=past_length,
            device=input_ids.device,
        )
        try:
            hidden_states = self.model(
                input_ids,
                position_ids=position_ids,
                layer_caches=cache.layers,
            )
            expected_length = past_length + sequence_length
            if cache.length != expected_length:
                raise RuntimeError(
                    f"KV cache length should be {expected_length}, "
                    f"got {cache.length}"
                )
            return self._project_logits(hidden_states)
        except Exception:
            cache.rollback(past_length)
            raise

    def _validate_cache(self, cache: SequenceKVCache) -> None:
        parameter = self.model.embed_tokens.weight
        expected = (
            self.config.num_key_value_heads,
            self.config.head_dim,
            parameter.dtype,
            parameter.device,
        )
        actual = (
            cache.num_key_value_heads,
            cache.head_dim,
            cache.dtype,
            cache.device,
        )
        expected_layers = self.config.num_hidden_layers
        if actual != expected or len(cache.layers) != expected_layers:
            raise ValueError(
                "cache is incompatible with this model: expected "
                f"{expected_layers} layers and KV heads/head dim/dtype/device "
                f"{expected}, got {len(cache.layers)} layers and {actual}"
            )

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
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_config: DecoderConfig | None = None,
    ) -> Self:
        """Build on meta, strictly assign one checkpoint, and return an eval model."""

        config = (
            cls.config_class.from_model_dir(model_dir)
            if model_config is None
            else model_config
        )
        if not isinstance(config, cls.config_class):
            raise TypeError(
                f"{cls.__name__} requires {cls.config_class.__name__}, "
                f"got {type(config).__name__}"
            )
        checkpoint = SafeTensorCheckpoint.from_model_dir(model_dir)
        with torch.device("meta"):
            model = cls(config)
        model.load_checkpoint(checkpoint)
        model.eval()
        return model
