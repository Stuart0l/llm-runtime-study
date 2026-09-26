"""Staggered-arrival serving benchmark: continuous batching versus serial."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Sequence

from mini_llm.engine import Engine
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage

from benchmarks.common import PromptCase


HEADERS = (
    "device",
    "mode",
    "requests",
    "interval",
    "max batch",
    "prompt/request",
    "makespan",
    "output tok/s",
    "TTFT p50",
    "TTFT p95",
    "latency p50",
    "latency p95",
)

MODES = ("continuous", "serial")
_GREEDY = SamplingConfig(temperature=0)


@dataclass(frozen=True, slots=True)
class _Run:
    """Per-request timings, in seconds from the first arrival."""

    arrivals: tuple[float, ...]
    first_tokens: tuple[float, ...]
    finishes: tuple[float, ...]
    generated_tokens: int

    @property
    def makespan(self) -> float:
        return max(self.finishes)

    @property
    def ttfts(self) -> list[float]:
        return [first - arrival for first, arrival in zip(self.first_tokens, self.arrivals)]

    @property
    def latencies(self) -> list[float]:
        return [finish - arrival for finish, arrival in zip(self.finishes, self.arrivals)]


class _Recorder:
    """Turn streamed events into per-request first-token and finish times."""

    def __init__(self, arrivals: Sequence[float], started: float) -> None:
        self.arrivals = tuple(arrivals)
        self.started = started
        self.first_tokens: list[float | None] = [None] * len(arrivals)
        self.finishes: list[float | None] = [None] * len(arrivals)
        self.generated_tokens = 0

    def now(self) -> float:
        return time.perf_counter() - self.started

    def wait_until(self, offset: float) -> None:
        delay = offset - self.now()
        if delay > 0:
            time.sleep(delay)

    def record(self, request: int, token_id: int | None, finished: bool) -> None:
        # Events are recorded when the host receives them, which is what a
        # client would observe; the step that produced them has completed.
        now = self.now()
        if self.first_tokens[request] is None:
            self.first_tokens[request] = now
        if token_id is not None:
            self.generated_tokens += 1
        if finished:
            self.finishes[request] = now

    def result(self) -> _Run:
        if any(value is None for value in (*self.first_tokens, *self.finishes)):
            raise RuntimeError("a benchmark request did not finish")
        return _Run(
            arrivals=self.arrivals,
            first_tokens=tuple(self.first_tokens),  # type: ignore[arg-type]
            finishes=tuple(self.finishes),  # type: ignore[arg-type]
            generated_tokens=self.generated_tokens,
        )


def _run_continuous(
    engine: Engine, messages: list[ChatMessage], arrivals: Sequence[float], decode_tokens: int
) -> _Run:
    scheduler = engine.create_scheduler()
    recorder = _Recorder(arrivals, time.perf_counter())
    pending = deque(enumerate(arrivals))
    request_for_index: dict[int, int] = {}
    try:
        while pending or scheduler.has_unfinished:
            # Submit every request whose arrival time has passed; they join
            # the running batch at the next step.
            while pending and pending[0][1] <= recorder.now():
                request, _ = pending.popleft()
                index = scheduler.add_request(
                    messages, max_new_tokens=decode_tokens, sampling=_GREEDY
                )
                request_for_index[index] = request
            if not scheduler.has_unfinished:
                # Idle until the next arrival instead of spinning.
                recorder.wait_until(pending[0][1])
                continue
            for index, event in scheduler.step():
                recorder.record(
                    request_for_index[index],
                    event.token_id,
                    event.finish_reason is not None,
                )
    finally:
        scheduler.abort_all()
    return recorder.result()


def _run_serial(
    engine: Engine, messages: list[ChatMessage], arrivals: Sequence[float], decode_tokens: int
) -> _Run:
    recorder = _Recorder(arrivals, time.perf_counter())
    # One request at a time in arrival order: a request that arrives while
    # another is generating waits in line, as with a serial server.
    for request, arrival in enumerate(arrivals):
        recorder.wait_until(arrival)
        for _, event in engine.generate(
            (messages,), max_new_tokens=decode_tokens, sampling=_GREEDY
        ):
            recorder.record(request, event.token_id, event.finish_reason is not None)
    return recorder.result()


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile."""

    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def run(
    engine: Engine,
    cases: Sequence[PromptCase],
    *,
    warmups: int,
    repeats: int,
    decode_tokens: int,
    requests: int,
    interval_seconds: float,
) -> list[tuple[str, ...]]:
    """Replay evenly spaced arrivals through both serving modes."""

    if requests <= 0:
        raise ValueError(f"requests must be positive, got {requests}")
    if interval_seconds < 0:
        raise ValueError(f"arrival interval must be non-negative, got {interval_seconds}")
    arrivals = tuple(index * interval_seconds for index in range(requests))
    runners = {"continuous": _run_continuous, "serial": _run_serial}

    rows: list[tuple[str, ...]] = []
    for case in cases:
        messages = [ChatMessage("user", case.prompt)]
        for mode in MODES:
            runner = runners[mode]
            # Warm-ups also capture every decode graph size the replay needs.
            for _ in range(warmups):
                runner(engine, messages, arrivals, decode_tokens)
            runs = [
                runner(engine, messages, arrivals, decode_tokens) for _ in range(repeats)
            ]
            makespan = median(run.makespan for run in runs)
            throughput = median(run.generated_tokens / run.makespan for run in runs)

            def stat(values: str, fraction: float) -> str:
                seconds = median(
                    _percentile(getattr(run, values), fraction) for run in runs
                )
                return f"{seconds * 1_000:.1f} ms"

            rows.append(
                (
                    engine.device.type,
                    mode,
                    str(requests),
                    f"{interval_seconds * 1_000:.0f} ms",
                    str(engine.max_batch_size if mode == "continuous" else 1),
                    str(case.actual_tokens),
                    f"{makespan:.2f} s",
                    f"{throughput:.1f}",
                    stat("ttfts", 0.5),
                    stat("ttfts", 0.95),
                    stat("latencies", 0.5),
                    stat("latencies", 0.95),
                )
            )
    return rows
