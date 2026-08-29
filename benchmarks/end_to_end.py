"""End-to-end greedy generation benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import time
from typing import Sequence

from mini_llm.engine import Engine
from mini_llm.interfaces import ChatMessage
from mini_llm.sampling import SamplingConfig

from benchmarks.common import PromptCase


HEADERS = (
    "device",
    "prompt",
    "TTFT",
    "prefill tok/s",
    "decode TPOT",
    "decode tok/s",
    "generated",
    "cache",
)


@dataclass(frozen=True, slots=True)
class _Run:
    prompt_tokens: int
    generated_tokens: int
    ttft: float
    prefill_seconds: float
    decode_seconds: float


def _run_once(engine: Engine, prompt: str, decode_tokens: int) -> _Run:
    stream = engine.generate(
        [ChatMessage("user", prompt)],
        max_new_tokens=decode_tokens,
        sampling=SamplingConfig(temperature=0),
    )
    started = time.perf_counter()
    first = next(stream)
    first_finished = time.perf_counter()
    events = [first]
    events.extend(stream)
    prompt_tokens = first.prompt_token_count
    if prompt_tokens is None or first.model_seconds is None:
        raise RuntimeError("generation did not report synchronized prefill metrics")
    return _Run(
        prompt_tokens=prompt_tokens,
        generated_tokens=sum(event.token_id is not None for event in events),
        ttft=first_finished - started,
        prefill_seconds=first.model_seconds,
        decode_seconds=sum(event.model_seconds or 0.0 for event in events[1:]),
    )


def run(
    engine: Engine,
    cases: Sequence[PromptCase],
    *,
    warmups: int,
    repeats: int,
    decode_tokens: int,
) -> list[tuple[str, ...]]:
    """Benchmark normal Engine generation without loading another model."""

    rows: list[tuple[str, ...]] = []
    for case in cases:
        for _ in range(warmups):
            _run_once(engine, case.prompt, decode_tokens)
        runs = tuple(
            _run_once(engine, case.prompt, decode_tokens)
            for _ in range(repeats)
        )
        prompt_tokens = runs[0].prompt_tokens
        generated_tokens = runs[0].generated_tokens
        ttft = median(run.ttft for run in runs)
        prefill_seconds = median(run.prefill_seconds for run in runs)
        decode_counts = [max(0, run.generated_tokens - 1) for run in runs]
        decode_tpots = [
            run.decode_seconds / count
            for run, count in zip(runs, decode_counts)
            if count > 0 and run.decode_seconds > 0
        ]
        decode_tpot = median(decode_tpots) if decode_tpots else None
        cache = engine.model.cache
        assert cache is not None
        rows.append(
            (
                engine.device.type,
                str(prompt_tokens),
                f"{ttft * 1_000:.2f} ms",
                f"{prompt_tokens / prefill_seconds:.2f}",
                "n/a" if decode_tpot is None else f"{decode_tpot * 1_000:.2f} ms",
                "n/a" if decode_tpot is None else f"{1 / decode_tpot:.2f}",
                str(generated_tokens),
                f"{cache.num_bytes / (1024**2):.2f} MiB",
            )
        )
    return rows
