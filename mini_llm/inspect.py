"""Command-line model configuration inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mini_llm.config import ConfigError, Qwen3Config


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a local Qwen3 model configuration."
    )
    parser.add_argument("model_dir", type=Path, help="directory containing config.json")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="KV-cache capacity used for the memory estimate (default: 4096)",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="KV-cache scalar type used for the estimate (default: float16)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size used for the cache estimate (default: 1)",
    )
    return parser


def render_summary(
    config: Qwen3Config,
    *,
    model_dir: Path,
    max_seq_len: int,
    cache_dtype: str,
    batch_size: int,
) -> str:
    cache_bytes = config.kv_cache_bytes(
        max_seq_len, batch_size=batch_size, dtype=cache_dtype
    )
    architecture = ", ".join(config.architectures) or "(not declared)"
    eos = ", ".join(str(item) for item in config.eos_token_ids)
    lines = [
        "Qwen3 configuration: valid",
        f"  model directory:       {model_dir}",
        f"  architecture:          {architecture}",
        f"  checkpoint dtype:      {config.torch_dtype}",
        f"  vocabulary:            {config.vocab_size:,}",
        f"  decoder layers:        {config.num_hidden_layers}",
        f"  hidden / MLP width:    {config.hidden_size:,} / {config.intermediate_size:,}",
        f"  query / KV heads:      {config.num_attention_heads} / {config.num_key_value_heads}",
        f"  head dimension:        {config.head_dim}",
        f"  query projection:      {config.query_projection_size:,}",
        f"  key/value projection:  {config.kv_projection_size:,} each",
        f"  queries per KV head:   {config.queries_per_kv_head}",
        f"  maximum positions:     {config.max_position_embeddings:,}",
        f"  RoPE theta:            {config.rope_theta:g}",
        f"  BOS / EOS token IDs:   {config.bos_token_id} / {eos}",
        "KV-cache estimate:",
        f"  capacity / batch:      {max_seq_len:,} / {batch_size}",
        f"  dtype:                 {cache_dtype}",
        f"  total:                 {_format_bytes(cache_bytes)} ({cache_bytes:,} bytes)",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = Qwen3Config.from_model_dir(args.model_dir)
        summary = render_summary(
            config,
            model_dir=args.model_dir,
            max_seq_len=args.max_seq_len,
            cache_dtype=args.cache_dtype,
            batch_size=args.batch_size,
        )
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
