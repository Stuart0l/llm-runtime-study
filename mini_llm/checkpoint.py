"""Lazy Safetensors checkpoint access and strict Qwen3 schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Iterable, Mapping

from safetensors import SafetensorError, safe_open
import torch

from mini_llm.config import Qwen3Config


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be read or does not match the model."""


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint contents do not match the expected tensor schema."""


_SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

_CONFIG_TO_SAFETENSOR_DTYPE = {
    "float16": "F16",
    "bfloat16": "BF16",
    "float32": "F32",
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Header information for one serialized tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    num_elements: int
    num_bytes: int


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Expected shape and dtype for one model tensor."""

    shape: tuple[int, ...]
    dtype: str


class SafeTensorCheckpoint:
    """Read a single-file checkpoint lazily.

    Construction reads the small Safetensors header and caches its manifest.
    Tensor payloads remain unmapped until :meth:`get_tensor` is called.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise CheckpointError(f"checkpoint file does not exist: {self.path}")
        if self.path.suffix != ".safetensors":
            raise CheckpointError(
                f"checkpoint must use the .safetensors extension: {self.path}"
            )

        try:
            with safe_open(self.path, framework="pt", device="cpu") as handle:
                self._metadata = dict(handle.metadata() or {})
                manifest = []
                for name in handle.keys():
                    view = handle.get_slice(name)
                    shape = tuple(view.get_shape())
                    dtype = view.get_dtype()
                    if dtype not in _SAFETENSOR_DTYPE_BYTES:
                        raise CheckpointError(
                            f"unsupported Safetensors dtype {dtype!r} for tensor {name!r}"
                        )
                    num_elements = reduce(mul, shape, 1)
                    manifest.append(
                        TensorInfo(
                            name=name,
                            shape=shape,
                            dtype=dtype,
                            num_elements=num_elements,
                            num_bytes=num_elements * _SAFETENSOR_DTYPE_BYTES[dtype],
                        )
                    )
        except SafetensorError as exc:
            raise CheckpointError(f"invalid Safetensors checkpoint: {self.path}") from exc

        self._manifest = tuple(manifest)
        self._by_name = {tensor.name: tensor for tensor in self._manifest}

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "SafeTensorCheckpoint":
        """Open the v1 single-file checkpoint from a local model directory."""

        model_path = Path(model_dir)
        checkpoint_path = model_path / "model.safetensors"
        if checkpoint_path.is_file():
            return cls(checkpoint_path)
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            raise CheckpointError(
                "sharded Safetensors checkpoints are not supported in v1: "
                f"{index_path}"
            )
        raise CheckpointError(f"checkpoint file does not exist: {checkpoint_path}")

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata.copy()

    @property
    def manifest(self) -> tuple[TensorInfo, ...]:
        return self._manifest

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self._manifest)

    @property
    def tensor_count(self) -> int:
        return len(self._manifest)

    @property
    def tensor_bytes(self) -> int:
        """Logical bytes occupied by tensor payloads, excluding file headers."""

        return sum(tensor.num_bytes for tensor in self._manifest)

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise CheckpointError(f"tensor is not present in checkpoint: {name!r}") from exc

    def get_tensor(self, name: str) -> torch.Tensor:
        """Materialize one named tensor on CPU without loading other payloads."""

        self.tensor_info(name)
        try:
            with safe_open(self.path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)
        except SafetensorError as exc:
            raise CheckpointError(f"could not load tensor {name!r}") from exc

    def get_tensors(self, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        """Materialize selected tensors on CPU while opening the file only once."""

        selected = self.tensor_names if names is None else tuple(names)
        for name in selected:
            self.tensor_info(name)
        try:
            with safe_open(self.path, framework="pt", device="cpu") as handle:
                return {name: handle.get_tensor(name) for name in selected}
        except SafetensorError as exc:
            raise CheckpointError("could not load checkpoint tensors") from exc


def expected_qwen3_tensors(config: Qwen3Config) -> dict[str, TensorSpec]:
    """Build the exact tensor contract for the supported dense Qwen3 model."""

    dtype = _CONFIG_TO_SAFETENSOR_DTYPE[config.torch_dtype]
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    query = config.query_projection_size
    key_value = config.kv_projection_size
    vocab = config.vocab_size

    specs = {
        "model.embed_tokens.weight": TensorSpec((vocab, hidden), dtype),
        "model.norm.weight": TensorSpec((hidden,), dtype),
        "lm_head.weight": TensorSpec((vocab, hidden), dtype),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        specs.update(
            {
                f"{prefix}.input_layernorm.weight": TensorSpec((hidden,), dtype),
                f"{prefix}.post_attention_layernorm.weight": TensorSpec(
                    (hidden,), dtype
                ),
                f"{prefix}.self_attn.q_norm.weight": TensorSpec(
                    (config.head_dim,), dtype
                ),
                f"{prefix}.self_attn.k_norm.weight": TensorSpec(
                    (config.head_dim,), dtype
                ),
                f"{prefix}.self_attn.q_proj.weight": TensorSpec(
                    (query, hidden), dtype
                ),
                f"{prefix}.self_attn.k_proj.weight": TensorSpec(
                    (key_value, hidden), dtype
                ),
                f"{prefix}.self_attn.v_proj.weight": TensorSpec(
                    (key_value, hidden), dtype
                ),
                f"{prefix}.self_attn.o_proj.weight": TensorSpec(
                    (hidden, query), dtype
                ),
                f"{prefix}.mlp.gate_proj.weight": TensorSpec(
                    (intermediate, hidden), dtype
                ),
                f"{prefix}.mlp.up_proj.weight": TensorSpec(
                    (intermediate, hidden), dtype
                ),
                f"{prefix}.mlp.down_proj.weight": TensorSpec(
                    (hidden, intermediate), dtype
                ),
            }
        )
        if config.attention_bias:
            specs.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": TensorSpec((query,), dtype),
                    f"{prefix}.self_attn.k_proj.bias": TensorSpec(
                        (key_value,), dtype
                    ),
                    f"{prefix}.self_attn.v_proj.bias": TensorSpec(
                        (key_value,), dtype
                    ),
                }
            )
    return specs


def validate_qwen3_checkpoint(
    checkpoint: SafeTensorCheckpoint, config: Qwen3Config
) -> None:
    """Validate all names, shapes, and dtypes, reporting every discovered issue."""

    expected = expected_qwen3_tensors(config)
    actual = {tensor.name: tensor for tensor in checkpoint.manifest}
    errors = []

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    if missing:
        errors.append("missing tensors: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected tensors: " + ", ".join(unexpected))

    for name in sorted(expected.keys() & actual.keys()):
        spec = expected[name]
        info = actual[name]
        if info.shape != spec.shape:
            errors.append(
                f"shape mismatch for {name}: expected {spec.shape}, got {info.shape}"
            )
        if info.dtype != spec.dtype:
            errors.append(
                f"dtype mismatch for {name}: expected {spec.dtype}, got {info.dtype}"
            )

    if errors:
        details = "\n  - ".join(errors)
        raise CheckpointValidationError(
            f"checkpoint does not match Qwen3 configuration:\n  - {details}"
        )
