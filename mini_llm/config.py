"""Typed configuration for the Qwen3 architecture.

This module deliberately has no PyTorch dependency.  Model metadata should be
inspectable before allocating weights or selecting an execution device.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


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


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """The Qwen3 fields needed to construct and run the transformer."""

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
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3Config":
        """Load and validate ``config.json`` from a local model directory."""

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
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Qwen3Config":
        """Construct a validated configuration from decoded JSON data."""

        architectures_raw = raw.get("architectures", [])
        if not isinstance(architectures_raw, list) or not all(
            isinstance(item, str) for item in architectures_raw
        ):
            raise ConfigError("architectures must be a list of strings")

        eos_raw = raw.get("eos_token_id")
        if isinstance(eos_raw, int) and not isinstance(eos_raw, bool):
            eos_token_ids = (eos_raw,)
        elif isinstance(eos_raw, list) and eos_raw and all(
            isinstance(item, int) and not isinstance(item, bool) for item in eos_raw
        ):
            eos_token_ids = tuple(eos_raw)
        else:
            raise ConfigError("eos_token_id must be an integer or a non-empty integer list")

        config = cls(
            architectures=tuple(architectures_raw),
            model_type=_required(raw, "model_type", str),
            vocab_size=_required(raw, "vocab_size", int),
            hidden_size=_required(raw, "hidden_size", int),
            intermediate_size=_required(raw, "intermediate_size", int),
            num_hidden_layers=_required(raw, "num_hidden_layers", int),
            num_attention_heads=_required(raw, "num_attention_heads", int),
            num_key_value_heads=_required(raw, "num_key_value_heads", int),
            head_dim=_required(raw, "head_dim", int),
            max_position_embeddings=_required(raw, "max_position_embeddings", int),
            rms_norm_eps=float(_required(raw, "rms_norm_eps", (int, float))),
            rope_theta=float(_required(raw, "rope_theta", (int, float))),
            hidden_act=_required(raw, "hidden_act", str),
            attention_bias=raw.get("attention_bias", False),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            torch_dtype=_required(raw, "torch_dtype", str),
            bos_token_id=_required(raw, "bos_token_id", int),
            eos_token_ids=eos_token_ids,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject unsupported or inconsistent architecture configurations."""

        if self.model_type != "qwen3":
            raise ConfigError(
                f"unsupported model_type {self.model_type!r}; this runtime supports 'qwen3'"
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
                f"unsupported hidden_act {self.hidden_act!r}; Qwen3 requires 'silu'"
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
            raise ConfigError(f"unsupported cache dtype {dtype!r}; expected one of: {supported}")
        return (
            self.num_hidden_layers
            * 2
            * batch_size
            * self.num_key_value_heads
            * max_seq_len
            * self.head_dim
            * _DTYPE_BYTES[dtype]
        )
