"""Typed configuration and architecture dispatch for supported models.

This module deliberately has no PyTorch dependency.  Model metadata should be
inspectable before allocating weights or selecting an execution device.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Self


class ConfigError(ValueError):
    """Raised when a model configuration is missing or internally inconsistent."""


_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def _required(
    config: Mapping[str, Any], name: str, expected_type: type | tuple[type, ...]
) -> Any:
    if name not in config:
        raise ConfigError(f"missing required configuration field: {name}")
    value = config[name]
    if not isinstance(value, expected_type) or isinstance(value, bool):
        expected_types = (
            expected_type if isinstance(expected_type, tuple) else (expected_type,)
        )
        expected_name = " or ".join(item.__name__ for item in expected_types)
        raise ConfigError(
            f"{name} must be {expected_name}, got {type(value).__name__}"
        )
    return value


def _load_config_dict(model_dir: str | Path) -> dict[str, Any]:
    """Read one model directory's JSON configuration without choosing a family."""

    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not model_path.is_dir():
        raise ConfigError(f"model directory does not exist: {model_path}")
    if not config_path.is_file():
        raise ConfigError(f"model configuration does not exist: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration root must be an object: {config_path}")
    return raw


def _eos_token_ids(raw: Mapping[str, Any]) -> tuple[int, ...]:
    eos_raw = raw.get("eos_token_id")
    if isinstance(eos_raw, int) and not isinstance(eos_raw, bool):
        return (eos_raw,)
    if isinstance(eos_raw, list) and eos_raw and all(
        isinstance(item, int) and not isinstance(item, bool) for item in eos_raw
    ):
        return tuple(eos_raw)
    raise ConfigError("eos_token_id must be an integer or a non-empty integer list")


def _common_decoder_fields(
    raw: Mapping[str, Any], *, head_dim: int | None = None
) -> dict[str, Any]:
    """Parse fields represented identically by all supported decoders.

    Granite omits ``head_dim`` from its JSON because it is derived from the
    hidden width and query-head count. Qwen stores it explicitly because its
    query projection can be wider than the residual stream.
    """

    architectures_raw = raw.get("architectures", [])
    if not isinstance(architectures_raw, list) or not all(
        isinstance(item, str) for item in architectures_raw
    ):
        raise ConfigError("architectures must be a list of strings")

    hidden_size = _required(raw, "hidden_size", int)
    num_attention_heads = _required(raw, "num_attention_heads", int)
    if head_dim is None:
        if num_attention_heads <= 0:
            raise ConfigError(
                "num_attention_heads must be positive before deriving head_dim"
            )
        head_dim, remainder = divmod(hidden_size, num_attention_heads)
        if remainder:
            raise ConfigError(
                "hidden_size must be divisible by num_attention_heads, got "
                f"{hidden_size} and {num_attention_heads}"
            )

    return {
        "architectures": tuple(architectures_raw),
        "model_type": _required(raw, "model_type", str),
        "vocab_size": _required(raw, "vocab_size", int),
        "hidden_size": hidden_size,
        "intermediate_size": _required(raw, "intermediate_size", int),
        "num_hidden_layers": _required(raw, "num_hidden_layers", int),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": _required(raw, "num_key_value_heads", int),
        "head_dim": head_dim,
        "max_position_embeddings": _required(raw, "max_position_embeddings", int),
        "rms_norm_eps": float(_required(raw, "rms_norm_eps", (int, float))),
        "rope_theta": float(_required(raw, "rope_theta", (int, float))),
        "hidden_act": _required(raw, "hidden_act", str),
        "attention_bias": raw.get("attention_bias", False),
        "attention_dropout": float(raw.get("attention_dropout", 0.0)),
        "tie_word_embeddings": raw.get("tie_word_embeddings", False),
        "torch_dtype": _required(raw, "torch_dtype", str),
        "bos_token_id": _required(raw, "bos_token_id", int),
        "eos_token_ids": _eos_token_ids(raw),
    }


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Fields and derived dimensions shared by decoder-only architectures."""

    architectures: tuple[str, ...]
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    tie_word_embeddings: bool
    torch_dtype: str
    bos_token_id: int
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> Self:
        """Load and validate ``config.json`` from a local model directory."""

        return cls.from_dict(_load_config_dict(model_dir))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        raise NotImplementedError

    def _validate_common(
        self,
        *,
        expected_model_type: str,
        expected_architecture: str,
        additional_token_ids: tuple[tuple[str, int], ...] = (),
    ) -> None:
        """Validate invariants shared by all currently supported decoders."""

        if self.model_type != expected_model_type:
            raise ConfigError(
                f"unsupported model_type {self.model_type!r}; "
                f"expected {expected_model_type!r}"
            )
        if self.architectures != (expected_architecture,):
            raise ConfigError(
                f"architecture must be {expected_architecture}, got {self.architectures}"
            )

        positive_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"for grouped-query attention, got {self.num_attention_heads} and "
                f"{self.num_key_value_heads}"
            )
        if self.head_dim % 2 != 0:
            raise ConfigError(
                f"head_dim must be even for rotary embeddings, got {self.head_dim}"
            )
        if self.rms_norm_eps <= 0:
            raise ConfigError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ConfigError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.hidden_act != "silu":
            raise ConfigError(
                f"unsupported hidden_act {self.hidden_act!r}; expected 'silu'"
            )
        if not isinstance(self.attention_bias, bool):
            raise ConfigError("attention_bias must be boolean")
        if self.attention_dropout < 0 or self.attention_dropout >= 1:
            raise ConfigError("attention_dropout must satisfy 0 <= value < 1")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ConfigError("tie_word_embeddings must be boolean")
        if self.torch_dtype not in _DTYPE_BYTES:
            supported = ", ".join(sorted(_DTYPE_BYTES))
            raise ConfigError(
                f"unsupported torch_dtype {self.torch_dtype!r}; expected one of: {supported}"
            )
        for name, token_id in (
            ("bos_token_id", self.bos_token_id),
            *(("eos_token_id", item) for item in self.eos_token_ids),
            *additional_token_ids,
        ):
            if not 0 <= token_id < self.vocab_size:
                raise ConfigError(
                    f"{name} must be within vocabulary [0, {self.vocab_size}), got {token_id}"
                )

    @property
    def query_projection_size(self) -> int:
        """Output width of the query projection."""

        return self.num_attention_heads * self.head_dim

    @property
    def kv_projection_size(self) -> int:
        """Output width of each key and value projection."""

        return self.num_key_value_heads * self.head_dim

    @property
    def queries_per_kv_head(self) -> int:
        """Number of query heads sharing one key/value head."""

        return self.num_attention_heads // self.num_key_value_heads

    def validate_context_length(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ConfigError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len > self.max_position_embeddings:
            raise ConfigError(
                f"max_seq_len {max_seq_len} exceeds the model limit "
                f"{self.max_position_embeddings}"
            )

    def kv_cache_bytes(
        self,
        max_seq_len: int,
        *,
        batch_size: int = 1,
        dtype: str = "float16",
    ) -> int:
        """Return bytes required by the dense K/V cache for all layers.

        Formula: layers * 2(K and V) * batch * KV heads * positions *
        head dimension * bytes per scalar.
        """

        self.validate_context_length(max_seq_len)
        if batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {batch_size}")
        if dtype not in _DTYPE_BYTES:
            supported = ", ".join(sorted(_DTYPE_BYTES))
            raise ConfigError(
                f"unsupported cache dtype {dtype!r}; expected one of: {supported}"
            )
        return (
            self.num_hidden_layers
            * 2
            * batch_size
            * self.num_key_value_heads
            * max_seq_len
            * self.head_dim
            * _DTYPE_BYTES[dtype]
        )


@dataclass(frozen=True, slots=True)
class Qwen3Config(DecoderConfig):
    """Qwen3 fields needed to construct and run the transformer."""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        config = cls(
            **_common_decoder_fields(raw, head_dim=_required(raw, "head_dim", int))
        )
        config.validate()
        return config

    def validate(self) -> None:
        self._validate_common(
            expected_model_type="qwen3",
            expected_architecture="Qwen3ForCausalLM",
        )


@dataclass(frozen=True, slots=True)
class GraniteMoeConfig(DecoderConfig):
    """Granite 3.1 1B-A400M fields needed for inference."""

    attention_multiplier: float
    embedding_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    num_local_experts: int
    num_experts_per_tok: int
    pad_token_id: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        config = cls(
            **_common_decoder_fields(raw),
            attention_multiplier=float(
                _required(raw, "attention_multiplier", (int, float))
            ),
            embedding_multiplier=float(
                _required(raw, "embedding_multiplier", (int, float))
            ),
            residual_multiplier=float(
                _required(raw, "residual_multiplier", (int, float))
            ),
            logits_scaling=float(_required(raw, "logits_scaling", (int, float))),
            num_local_experts=_required(raw, "num_local_experts", int),
            num_experts_per_tok=_required(raw, "num_experts_per_tok", int),
            pad_token_id=_required(raw, "pad_token_id", int),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self._validate_common(
            expected_model_type="granitemoe",
            expected_architecture="GraniteMoeForCausalLM",
            additional_token_ids=(("pad_token_id", self.pad_token_id),),
        )
        positive_fields = {
            "num_local_experts": self.num_local_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")

        if self.num_experts_per_tok > self.num_local_experts:
            raise ConfigError(
                "num_experts_per_tok cannot exceed num_local_experts, got "
                f"{self.num_experts_per_tok} and {self.num_local_experts}"
            )
        for name, value in (
            ("attention_multiplier", self.attention_multiplier),
            ("embedding_multiplier", self.embedding_multiplier),
            ("residual_multiplier", self.residual_multiplier),
            ("logits_scaling", self.logits_scaling),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        if not self.tie_word_embeddings:
            raise ConfigError("Granite 3.1 requires tied word embeddings")

    @property
    def parameters_per_expert(self) -> int:
        """Packed SwiGLU input and output weights for one expert."""

        return 3 * self.hidden_size * self.intermediate_size

    @property
    def total_expert_parameters(self) -> int:
        return (
            self.num_hidden_layers
            * self.num_local_experts
            * self.parameters_per_expert
        )

    @property
    def active_expert_parameters(self) -> int:
        return (
            self.num_hidden_layers
            * self.num_experts_per_tok
            * self.parameters_per_expert
        )

    @property
    def total_parameter_estimate(self) -> int:
        embedding = self.vocab_size * self.hidden_size
        attention_per_layer = (
            self.query_projection_size * self.hidden_size
            + 2 * self.kv_projection_size * self.hidden_size
            + self.hidden_size * self.query_projection_size
        )
        norms_per_layer = 2 * self.hidden_size
        router_per_layer = self.num_local_experts * self.hidden_size
        return (
            embedding
            + self.num_hidden_layers
            * (attention_per_layer + norms_per_layer + router_per_layer)
            + self.total_expert_parameters
            + self.hidden_size
        )

    @property
    def active_parameter_estimate(self) -> int:
        return self.total_parameter_estimate - self.total_expert_parameters + (
            self.active_expert_parameters
        )


ModelConfig = Qwen3Config | GraniteMoeConfig


def load_config(model_dir: str | Path) -> ModelConfig:
    """Load the appropriate typed configuration using ``model_type``."""

    raw = _load_config_dict(model_dir)
    model_type = raw.get("model_type")
    if model_type == "qwen3":
        return Qwen3Config.from_dict(raw)
    if model_type == "granitemoe":
        return GraniteMoeConfig.from_dict(raw)
    raise ConfigError(
        f"unsupported model_type {model_type!r}; expected 'qwen3' or 'granitemoe'"
    )
