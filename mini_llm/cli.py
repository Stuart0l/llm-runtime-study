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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mini_llm",
        description="Run the study-oriented Qwen3 inference runtime.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        help="raw prompt for one-shot mode, or an optional first interactive prompt",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="load the model once and repeatedly read prompts",
    )
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


def _sampling_from_args(args: argparse.Namespace) -> SamplingConfig:
    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )


def _engine_from_args(args: argparse.Namespace) -> Engine:
    return Engine.from_model_dir(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )


def run(args: argparse.Namespace, *, output: TextIO) -> RunMetrics:
    """Load an engine, generate one response, and return benchmark metrics."""

    if args.prompt is None:
        raise GenerationError("--prompt is required unless --interactive is used")
    sampling = _sampling_from_args(args)
    engine = _engine_from_args(args)
    return _run_prompt(
        engine,
        args.prompt,
        args,
        sampling=sampling,
        output=output,
    )


def _run_prompt(
    engine: Engine,
    prompt: str,
    args: argparse.Namespace,
    *,
    sampling: SamplingConfig,
    output: TextIO,
    include_load_time: bool = True,
) -> RunMetrics:
    """Generate one response using an already-loaded engine."""

    stream = engine.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
        enable_thinking=args.thinking,
    )
    generation_started = time.perf_counter()
    events: list[GenerationEvent] = []
    first_token_finished: float | None = None
    for event in stream:
        if first_token_finished is None:
            first_token_finished = time.perf_counter()
        events.append(event)
        if not args.no_stream and event.text_delta:
            print(event.text_delta, end="", flush=True, file=output)
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
        _print_metrics(
            engine,
            metrics,
            output=output,
            include_load_time=include_load_time,
        )
    return metrics


def run_interactive(
    args: argparse.Namespace,
    *,
    input_stream: TextIO,
    output: TextIO,
    error: TextIO,
) -> None:
    """Load one engine and serve independent prompts until the user exits."""

    sampling = _sampling_from_args(args)
    engine = _engine_from_args(args)
    print(
        f"Loaded model on {engine.device} with {engine.dtype} in "
        f"{engine.load_seconds:.2f} s.",
        file=output,
    )
    print("Enter a prompt. Use /quit or /exit to stop.", file=output)

    initial_prompt = args.prompt
    while True:
        if initial_prompt is not None:
            prompt = initial_prompt
            initial_prompt = None
            print(f"you> {prompt}", file=output)
        else:
            print("you> ", end="", flush=True, file=output)
            prompt = input_stream.readline()
            if prompt == "":
                print(file=output)
                return
            prompt = prompt.rstrip("\r\n")

        if prompt.strip() in {"/quit", "/exit"}:
            return
        if not prompt.strip():
            continue

        print("assistant> ", end="", flush=True, file=output)
        try:
            _run_prompt(
                engine,
                prompt,
                args,
                sampling=sampling,
                output=output,
                include_load_time=False,
            )
        except (GenerationError, KVCacheError) as exc:
            print(file=output)
            print(f"error: {exc}", file=error)


def _print_metrics(
    engine: Engine,
    metrics: RunMetrics,
    *,
    output: TextIO,
    include_load_time: bool = True,
) -> None:
    cache = engine.model.cache
    cache_text = "not allocated"
    if cache is not None:
        cache_text = f"{cache.num_bytes / (1024**2):.2f} MiB ({cache.capacity} positions)"
    decode_token_count = max(0, metrics.generated_tokens - 1)
    decode_rate = (
        decode_token_count / metrics.decode_seconds
        if decode_token_count and metrics.decode_seconds > 0
        else None
    )
    decode_text = "n/a" if decode_rate is None else f"{decode_rate:.2f} tokens/s"

    print("\nmetrics", file=output)
    print(f"  device:              {engine.device}", file=output)
    print(f"  dtype:               {engine.dtype}", file=output)
    if include_load_time:
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
    input_stream: TextIO | None = None,
    output: TextIO | None = None,
    error: TextIO | None = None,
) -> int:
    input_stream = sys.stdin if input_stream is None else input_stream
    output = sys.stdout if output is None else output
    error = sys.stderr if error is None else error
    args = build_parser().parse_args(argv)
    try:
        if args.interactive:
            run_interactive(
                args,
                input_stream=input_stream,
                output=output,
                error=error,
            )
        else:
            run(args, output=output)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=output)
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
