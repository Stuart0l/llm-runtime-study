"""Lazy Safetensors access and architecture-specific schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import json
from operator import mul
from pathlib import Path
from typing import Callable, Iterable, Mapping

from safetensors import SafetensorError, safe_open
import torch

from mini_llm.config import DecoderConfig, GraniteMoeConfig, Qwen3Config

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
    """Read a single-file or indexed sharded checkpoint lazily.

    Construction reads the small Safetensors headers and caches one combined
    manifest. Tensor payloads remain unmapped until :meth:`get_tensor` or
    :meth:`get_tensors` is called.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise CheckpointError(f"checkpoint file does not exist: {self.path}")
        if self.path.suffix != ".safetensors":
            raise CheckpointError(
                f"checkpoint must use the .safetensors extension: {self.path}"
            )

        manifest = self._inspect_shard(self.path)
        self._tensor_paths = {tensor.name: self.path for tensor in manifest}
        self._set_manifest(manifest)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "SafeTensorCheckpoint":
        """Open a single-file checkpoint or its Safetensors shard index."""

        model_path = Path(model_dir)
        checkpoint_path = model_path / "model.safetensors"
        if checkpoint_path.is_file():
            return cls(checkpoint_path)
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            return cls._from_index(index_path)
        raise CheckpointError(
            "model directory contains neither model.safetensors nor "
            f"model.safetensors.index.json: {model_path}"
        )

    @classmethod
    def _from_index(cls, index_path: Path) -> "SafeTensorCheckpoint":
        """Open a standard Hugging Face Safetensors shard index.

        The runtime assumes the index is trusted: every listed shard is in the
        model directory and every tensor-to-shard mapping is correct. The
        ordinary Qwen3 checkpoint validator still checks the combined tensor
        names, shapes, and dtypes before model loading.
        """

        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = raw["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CheckpointError(
                f"could not read checkpoint index {index_path}: {exc}"
            ) from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(
                f"checkpoint index weight_map must be a non-empty object: {index_path}"
            )

        model_dir = index_path.parent
        shard_paths = tuple(
            model_dir / filename for filename in dict.fromkeys(weight_map.values())
        )
        manifest = []
        for shard_path in shard_paths:
            if not shard_path.is_file():
                raise CheckpointError(f"checkpoint shard does not exist: {shard_path}")
            manifest.extend(cls._inspect_shard(shard_path))

        instance = cls.__new__(cls)
        instance.path = index_path
        instance._tensor_paths = {
            name: model_dir / filename for name, filename in weight_map.items()
        }
        instance._set_manifest(manifest)
        return instance

    @staticmethod
    def _inspect_shard(path: Path) -> list[TensorInfo]:
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
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
                return manifest
        except SafetensorError as exc:
            raise CheckpointError(f"invalid Safetensors checkpoint: {path}") from exc

    def _set_manifest(self, manifest: Iterable[TensorInfo]) -> None:
        self._manifest = tuple(sorted(manifest, key=lambda tensor: tensor.name))
        self._by_name = {tensor.name: tensor for tensor in self._manifest}

    @property
    def manifest(self) -> tuple[TensorInfo, ...]:
        return self._manifest

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise CheckpointError(f"tensor is not present in checkpoint: {name!r}") from exc

    def get_tensor(self, name: str) -> torch.Tensor:
        """Materialize one named tensor on CPU without loading other payloads."""

        self.tensor_info(name)
        tensor_path = self._tensor_paths[name]
        try:
            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(name)
        except SafetensorError as exc:
            raise CheckpointError(f"could not load tensor {name!r}") from exc

    def get_tensors(self, names: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        """Materialize selected tensors, opening each needed shard only once."""

        selected = tuple(self._by_name) if names is None else tuple(names)
        for name in selected:
            self.tensor_info(name)
        names_by_path: dict[Path, list[str]] = {}
        for name in selected:
            names_by_path.setdefault(self._tensor_paths[name], []).append(name)

        tensors: dict[str, torch.Tensor] = {}
        try:
            for path, shard_names in names_by_path.items():
                with safe_open(path, framework="pt", device="cpu") as handle:
                    tensors.update(
                        (name, handle.get_tensor(name)) for name in shard_names
                    )
        except SafetensorError as exc:
            raise CheckpointError("could not load checkpoint tensors") from exc
        return {name: tensors[name] for name in selected}

def _qwen3_projection_shapes(
    config: Qwen3Config,
) -> dict[str, tuple[int, int]]:
    """Return projection prefixes mapped to logical (input, output) widths."""

    per_layer = {
        "self_attn.q_proj": (config.hidden_size, config.query_projection_size),
        "self_attn.k_proj": (config.hidden_size, config.kv_projection_size),
        "self_attn.v_proj": (config.hidden_size, config.kv_projection_size),
        "self_attn.o_proj": (config.query_projection_size, config.hidden_size),
        "mlp.gate_proj": (config.hidden_size, config.intermediate_size),
        "mlp.up_proj": (config.hidden_size, config.intermediate_size),
        "mlp.down_proj": (config.intermediate_size, config.hidden_size),
    }
    return {
        f"model.layers.{layer}.{name}": shape
        for layer in range(config.num_hidden_layers)
        for name, shape in per_layer.items()
    }

def _gptq_projection_specs(
    prefix: str,
    *,
    in_features: int,
    out_features: int,
    bits: int,
    group_size: int,
    scale_dtype: str,
    bias: bool,
) -> dict[str, TensorSpec]:
    """Build the GPTQ v1 packed tensor contract for one linear projection."""

    pack_factor = 32 // bits
    invalid = []
    if in_features % pack_factor:
        invalid.append(f"input width {in_features} is not divisible by {pack_factor}")
    if out_features % pack_factor:
        invalid.append(
            f"output width {out_features} is not divisible by {pack_factor}"
        )
    if in_features % group_size:
        invalid.append(
            f"input width {in_features} is not divisible by group size {group_size}"
        )
    if invalid:
        raise CheckpointValidationError(
            f"projection {prefix} cannot use the configured GPTQ layout: "
            + "; ".join(invalid)
        )

    groups = in_features // group_size
    specs = {
        f"{prefix}.qweight": TensorSpec(
            (in_features // pack_factor, out_features), "I32"
        ),
        f"{prefix}.qzeros": TensorSpec(
            (groups, out_features // pack_factor), "I32"
        ),
        f"{prefix}.scales": TensorSpec((groups, out_features), scale_dtype),
        f"{prefix}.g_idx": TensorSpec((in_features,), "I32"),
    }
    if bias:
        specs[f"{prefix}.bias"] = TensorSpec((out_features,), scale_dtype)
    return specs

def _decoder_root_specs(config: DecoderConfig, dtype: str) -> dict[str, TensorSpec]:
    """Return tensor specs shared by supported decoder checkpoint layouts."""

    return {
        "model.embed_tokens.weight": TensorSpec(
            (config.vocab_size, config.hidden_size), dtype
        ),
        "model.norm.weight": TensorSpec((config.hidden_size,), dtype),
    }

def expected_qwen3_tensors(config: Qwen3Config) -> dict[str, TensorSpec]:
    """Build the Qwen logical schema in its declared dense or GPTQ layout."""

    dtype = _CONFIG_TO_SAFETENSOR_DTYPE[config.torch_dtype]
    quantization = config.quantization_config
    specs = _decoder_root_specs(config, dtype)
    layer_specs = {
        "input_layernorm.weight": TensorSpec((config.hidden_size,), dtype),
        "post_attention_layernorm.weight": TensorSpec((config.hidden_size,), dtype),
        "self_attn.q_norm.weight": TensorSpec((config.head_dim,), dtype),
        "self_attn.k_norm.weight": TensorSpec((config.head_dim,), dtype),
    }
    projection_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    projection_shapes = _qwen3_projection_shapes(config)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        specs.update(
            {f"{prefix}.{name}": spec for name, spec in layer_specs.items()}
        )
        for name in projection_names:
            projection = f"{prefix}.{name}"
            in_features, out_features = projection_shapes[projection]
            has_bias = config.attention_bias and name.endswith(
                ("q_proj", "k_proj", "v_proj")
            )
            if quantization is None:
                specs[f"{projection}.weight"] = TensorSpec(
                    (out_features, in_features), dtype
                )
                if has_bias:
                    specs[f"{projection}.bias"] = TensorSpec(
                        (out_features,), dtype
                    )
            else:
                specs.update(
                    _gptq_projection_specs(
                        projection,
                        in_features=in_features,
                        out_features=out_features,
                        bits=quantization.bits,
                        group_size=quantization.group_size,
                        scale_dtype="F16",
                        bias=has_bias,
                    )
                )
    if quantization is None:
        specs["lm_head.weight"] = TensorSpec(
            (config.vocab_size, config.hidden_size), dtype
        )
    return specs


def expected_granite_moe_tensors(
    config: GraniteMoeConfig,
) -> dict[str, TensorSpec]:
    """Build the Granite tied-embedding and packed-expert tensor contract."""

    dtype = _CONFIG_TO_SAFETENSOR_DTYPE[config.torch_dtype]
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    query = config.query_projection_size
    key_value = config.kv_projection_size
    experts = config.num_local_experts

    # Granite ties the output projection to its embedding matrix, so it has no
    # separate lm_head.weight tensor.
    specs = _decoder_root_specs(config, dtype)
    layer_specs = {
        "input_layernorm.weight": TensorSpec((hidden,), dtype),
        "post_attention_layernorm.weight": TensorSpec((hidden,), dtype),
        "self_attn.q_proj.weight": TensorSpec((query, hidden), dtype),
        "self_attn.k_proj.weight": TensorSpec((key_value, hidden), dtype),
        "self_attn.v_proj.weight": TensorSpec((key_value, hidden), dtype),
        "self_attn.o_proj.weight": TensorSpec((hidden, query), dtype),
        "block_sparse_moe.router.layer.weight": TensorSpec(
            (experts, hidden), dtype
        ),
        # Gate and up projections are packed together along dimension 1.
        "block_sparse_moe.input_linear.weight": TensorSpec(
            (experts, 2 * intermediate, hidden), dtype
        ),
        "block_sparse_moe.output_linear.weight": TensorSpec(
            (experts, hidden, intermediate), dtype
        ),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        specs.update(
            {f"{prefix}.{name}": spec for name, spec in layer_specs.items()}
        )
    return specs



CHECKPOINT_SCHEMA_BUILDERS: dict[
    str, Callable[..., dict[str, TensorSpec]]
] = {
    "qwen3": expected_qwen3_tensors,
    "granitemoe": expected_granite_moe_tensors,
}

def expected_model_tensors(config: DecoderConfig) -> dict[str, TensorSpec]:
    """Build the registered tensor schema for a model configuration."""

    try:
        builder = CHECKPOINT_SCHEMA_BUILDERS[config.model_type]
    except KeyError as exc:
        raise CheckpointValidationError(
            f"no checkpoint schema is registered for model_type {config.model_type!r}"
        ) from exc
    return builder(config)


def validate_checkpoint(
    checkpoint: SafeTensorCheckpoint, config: DecoderConfig
) -> None:
    """Validate all tensor names, shapes, and dtypes for a supported model."""

    expected = expected_model_tensors(config)
    _validate_tensor_manifest(
        checkpoint,
        expected,
        architecture=config.model_type,
    )
    if isinstance(config, Qwen3Config) and config.quantization_config is not None:
        _validate_qwen3_gptq_group_indices(checkpoint, config)

def _validate_qwen3_gptq_group_indices(
    checkpoint: SafeTensorCheckpoint, config: Qwen3Config
) -> None:
    """Require canonical non-activation-ordered GPTQ group assignments."""

    quantization = config.quantization_config
    if quantization is None:
        return
    projection_shapes = _qwen3_projection_shapes(config)
    names = tuple(f"{prefix}.g_idx" for prefix in projection_shapes)
    indices = checkpoint.get_tensors(names)
    errors = []
    for prefix, (in_features, _out_features) in projection_shapes.items():
        name = f"{prefix}.g_idx"
        actual = indices[name]
        expected = torch.arange(in_features, dtype=torch.int32).div(
            quantization.group_size, rounding_mode="floor"
        )
        mismatches = torch.nonzero(actual != expected, as_tuple=False)
        if mismatches.numel():
            position = int(mismatches[0, 0])
            errors.append(
                f"{name} at index {position}: expected "
                f"{int(expected[position])}, got {int(actual[position])}"
            )
    if errors:
        details = "\n  - ".join(errors)
        raise CheckpointValidationError(
            "checkpoint has invalid GPTQ group indices:\n  - " + details
        )

def _validate_tensor_manifest(
    checkpoint: SafeTensorCheckpoint,
    expected: Mapping[str, TensorSpec],
    *,
    architecture: str,
) -> None:
    """Compare one actual manifest with an already-built tensor contract."""

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
            f"checkpoint does not match {architecture} configuration:\n  - {details}"
        )
