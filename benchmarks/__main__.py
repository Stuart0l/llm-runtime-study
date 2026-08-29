"""Single-load command-line runner for all runtime benchmarks."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time
from typing import Sequence, TextIO

import torch

from benchmarks import cache_decode, end_to_end, moe_prefill
from benchmarks.common import PromptCase, build_prompt_case, render_table
from mini_llm.engine import Engine, EngineError


BENCHMARKS = ("cache-decode", "moe-prefill", "end-to-end")


class BenchmarkError(ValueError):
    """Raised for an invalid or unsupported benchmark request."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Benchmark resident mini-llm models without repeated loading.",
    )
    parser.add_argument(
        "--model", type=Path, action="append", required=True,
        help="model directory; repeat to benchmark multiple models",
    )
    parser.add_argument(
        "--benchmark",
        nargs="+",
        choices=BENCHMARKS,
        help="benchmark suites (default: all applicable suites)",
    )
    parser.add_argument(
        "--device",
        action="append",
        choices=("cpu", "mps"),
        help="device; repeat to select both (default: every available device)",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=_positive_int,
        default=[32, 128, 512],
    )
    parser.add_argument("--warmups", type=_non_negative_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--decode-tokens", type=_positive_int, default=16)
    return parser


def _selected_devices(requested: Sequence[str] | None) -> list[str]:
    if requested is None:
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices

    selected = set(requested)
    if "mps" in selected and not torch.backends.mps.is_available():
        raise BenchmarkError("MPS was requested but is unavailable")
    # CPU first allows one loaded model to move in a single direction.
    return [device for device in ("cpu", "mps") if device in selected]


def _validate_cases(
    engine: Engine, cases: Sequence[PromptCase], decode_tokens: int
) -> None:
    longest = max(case.actual_tokens for case in cases)
    required = longest + decode_tokens
    if required > engine.max_seq_len:
        raise BenchmarkError(
            f"longest prompt and decode require {required} positions, but the "
            f"benchmark engine limit is {engine.max_seq_len}"
        )


def _print_results(
    model_dir: Path,
    rows: dict[str, list[tuple[str, ...]]],
    *,
    output: TextIO,
) -> None:
    print(f"\n{model_dir.name}", file=output)
    for name in BENCHMARKS:
        suite_rows = rows.get(name)
        if not suite_rows:
            continue
        headers = {
            "cache-decode": cache_decode.HEADERS,
            "moe-prefill": moe_prefill.HEADERS,
            "end-to-end": end_to_end.HEADERS,
        }[name]
        print(f"\n{name}", file=output)
        print(render_table(headers, suite_rows), file=output)


def run(args: argparse.Namespace, *, output: TextIO) -> None:
    devices = _selected_devices(args.device)
    selected = list(dict.fromkeys(args.benchmark or BENCHMARKS))
    explicitly_selected = args.benchmark is not None

    for model_dir in args.model:
        first_device = devices[0]
        # Allow enough room for tokenization to land slightly above a requested
        # length while keeping cache allocations tied to actual requests.
        max_seq_len = max(args.prompt_lengths) + args.decode_tokens + 64
        engine = Engine.from_model_dir(
            model_dir,
            device=first_device,
            dtype="float16",
            max_seq_len=max_seq_len,
        )
        print(
            f"Loaded {model_dir.name} once on {first_device} in "
            f"{engine.load_seconds:.2f} s.",
            file=output,
        )
        cases = tuple(
            build_prompt_case(engine, length) for length in args.prompt_lengths
        )
        _validate_cases(engine, cases, args.decode_tokens)

        is_granite = engine.model.config.model_type == "granitemoe"
        if explicitly_selected and "moe-prefill" in selected and not is_granite:
            raise BenchmarkError(
                f"moe-prefill requires Granite MoE, got "
                f"{engine.model.config.model_type}"
            )

        rows: dict[str, list[tuple[str, ...]]] = {
            name: [] for name in selected
        }
        cpu_moe_references: dict[int, torch.Tensor] = {}
        for device_index, device in enumerate(devices):
            if device_index:
                started = time.perf_counter()
                engine.to(device=device)
                transfer_seconds = time.perf_counter() - started
                print(
                    f"Moved resident {model_dir.name} to {device} in "
                    f"{transfer_seconds:.2f} s without reloading.",
                    file=output,
                )

            if "cache-decode" in selected:
                rows["cache-decode"].extend(
                    cache_decode.run(
                        engine,
                        cases,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        decode_tokens=args.decode_tokens,
                    )
                )
            if "moe-prefill" in selected and is_granite:
                rows["moe-prefill"].extend(
                    moe_prefill.run(
                        engine,
                        cases,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        cpu_references=cpu_moe_references,
                    )
                )
            if "end-to-end" in selected:
                rows["end-to-end"].extend(
                    end_to_end.run(
                        engine,
                        cases,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        decode_tokens=args.decode_tokens,
                    )
                )

        _print_results(model_dir, rows, output=output)
        del engine
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    error: TextIO | None = None,
) -> int:
    output = sys.stdout if output is None else output
    error = sys.stderr if error is None else error
    args = build_parser().parse_args(argv)
    try:
        run(args, output=output)
    except (BenchmarkError, EngineError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

