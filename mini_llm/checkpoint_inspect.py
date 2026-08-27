"""Validate and inspect a local Qwen3 Safetensors checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from mini_llm.checkpoint import (
    CheckpointError,
    SafeTensorCheckpoint,
    validate_qwen3_checkpoint,
)
from mini_llm.config import ConfigError, Qwen3Config


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a Qwen3 Safetensors checkpoint."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--tensor",
        help="materialize one tensor and display its first values",
    )
    parser.add_argument(
        "--values",
        type=int,
        default=8,
        help="number of flattened tensor values to display (default: 8)",
    )
    return parser


def render_summary(
    checkpoint: SafeTensorCheckpoint, config: Qwen3Config
) -> str:
    dtypes = Counter(tensor.dtype for tensor in checkpoint.manifest)
    dtype_summary = ", ".join(
        f"{dtype}: {count}" for dtype, count in sorted(dtypes.items())
    )
    lines = [
        "Qwen3 checkpoint: valid",
        f"  file:                  {checkpoint.path}",
        f"  metadata:              {dict(checkpoint.metadata)}",
        f"  tensors:               {checkpoint.tensor_count}",
        f"  logical tensor bytes:  {_format_bytes(checkpoint.tensor_bytes)}",
        f"  dtypes:                {dtype_summary}",
        "Global tensors:",
    ]
    for name in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ):
        info = checkpoint.tensor_info(name)
        lines.append(f"  {name:<31} {str(info.shape):<22} {info.dtype}")
    lines.append("Decoder layers:")
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        tensors = [item for item in checkpoint.manifest if item.name.startswith(prefix)]
        size = sum(item.num_bytes for item in tensors)
        lines.append(
            f"  layer {layer:>2}: {len(tensors):>2} tensors, {_format_bytes(size):>10}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.values < 0:
        raise SystemExit("--values must be non-negative")
    try:
        config = Qwen3Config.from_model_dir(args.model_dir)
        checkpoint = SafeTensorCheckpoint.from_model_dir(args.model_dir)
        validate_qwen3_checkpoint(checkpoint, config)
    except (CheckpointError, ConfigError) as exc:
        raise SystemExit(f"checkpoint error: {exc}") from exc
    print(render_summary(checkpoint, config))

    if args.tensor:
        try:
            tensor = checkpoint.get_tensor(args.tensor)
        except CheckpointError as exc:
            raise SystemExit(f"checkpoint error: {exc}") from exc
        values = tensor.flatten()[: args.values].tolist()
        print("Selected tensor:")
        print(f"  name:    {args.tensor}")
        print(f"  shape:   {tuple(tensor.shape)}")
        print(f"  dtype:   {tensor.dtype}")
        print(f"  device:  {tensor.device}")
        print(f"  values:  {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
