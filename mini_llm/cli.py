"""Command-line generation and benchmark reporting for the mini runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Sequence, TextIO

from mini_llm.cache import KVCacheError
from mini_llm.checkpoint import CheckpointValidationError
from mini_llm.config import ConfigError
from mini_llm.engine import Engine, EngineError
from mini_llm.generation import GenerationError, GenerationEvent
from mini_llm.sampling import SamplingConfig, SamplingError
from mini_llm.tokenizer import TokenizerError


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Measurements collected from one streamed generation request."""

    prompt_tokens: int
    generated_tokens: int
    time_to_first_token: float
    decode_seconds: float
    finish_reason: str

    @property
    def decode_tokens_per_second(self) -> float | None:
        decode_tokens = max(0, self.generated_tokens - 1)
        if decode_tokens == 0 or self.decode_seconds <= 0:
            return None
        return decode_tokens / self.decode_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mini_llm",
        description="Run the study-oriented Qwen3 inference runtime.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    return parser


def run(args: argparse.Namespace, *, output: TextIO) -> RunMetrics:
    """Load an engine, stream one response, and return benchmark metrics."""

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    engine = Engine.from_model_dir(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )
    stream = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        enable_thinking=args.thinking,
    )
    engine.synchronize()
    generation_started = time.perf_counter()
    events: list[GenerationEvent] = []
    first_token_finished: float | None = None
    for event in stream:
        engine.synchronize()
        if first_token_finished is None:
            first_token_finished = time.perf_counter()
        events.append(event)
        if not args.no_stream and event.text_delta:
            print(event.text_delta, end="", flush=True, file=output)
    engine.synchronize()
    generation_finished = time.perf_counter()

    if not events:
        raise GenerationError("generation ended without a terminal event")
    prompt_tokens = events[0].prompt_token_count
    if prompt_tokens is None:
        raise GenerationError("first generation event is missing prompt-token count")
    if args.no_stream:
        print(events[-1].text, file=output)
    else:
        print(file=output)

    first_token_finished = first_token_finished or generation_finished
    generated_tokens = sum(event.token_id is not None for event in events)
    decode_seconds = sum(
        event.model_seconds or 0.0 for event in events[1:]
    )
    metrics = RunMetrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        time_to_first_token=first_token_finished - generation_started,
        decode_seconds=decode_seconds,
        finish_reason=events[-1].finish_reason or "unknown",
    )
    if not args.no_metrics:
        _print_metrics(engine, metrics, output=output)
    return metrics


def _print_metrics(engine: Engine, metrics: RunMetrics, *, output: TextIO) -> None:
    cache = engine.model.cache
    cache_text = "not allocated"
    if cache is not None:
        cache_text = f"{cache.num_bytes / (1024**2):.2f} MiB ({cache.capacity} positions)"
    decode_rate = metrics.decode_tokens_per_second
    decode_text = "n/a" if decode_rate is None else f"{decode_rate:.2f} tokens/s"

    print("\nmetrics", file=output)
    print(f"  device:              {engine.device}", file=output)
    print(f"  dtype:               {engine.dtype}", file=output)
    print(f"  load time:           {engine.load_seconds:.2f} s", file=output)
    print(f"  prompt tokens:       {metrics.prompt_tokens}", file=output)
    print(f"  generated tokens:    {metrics.generated_tokens}", file=output)
    print(f"  cache:               {cache_text}", file=output)
    print(
        f"  time to first token: {metrics.time_to_first_token * 1_000:.2f} ms",
        file=output,
    )
    print(f"  decode throughput:   {decode_text}", file=output)
    print(f"  stop reason:         {metrics.finish_reason}", file=output)


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
    except (
        CheckpointValidationError,
        ConfigError,
        EngineError,
        GenerationError,
        KVCacheError,
        SamplingError,
        TokenizerError,
    ) as exc:
        print(f"error: {exc}", file=error)
        return 2
    return 0
