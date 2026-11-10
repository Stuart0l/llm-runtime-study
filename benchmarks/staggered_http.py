"""Staggered-arrival benchmark against a running OpenAI-compatible HTTP server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from statistics import median
import time
from typing import Any, Sequence

from benchmarks.common import PromptCase, percentile


HEADERS = (
    "url",
    "requests",
    "interval",
    "max in-flight",
    "prompt/request",
    "makespan",
    "output tok/s",
    "latency p50",
    "latency p95",
)


class StaggeredHttpError(RuntimeError):
    """A staggered-http run failed; its numbers must not be recorded."""


@dataclass(frozen=True, slots=True)
class _Replay:
    """One replay's timings, in seconds from its start."""

    arrivals: tuple[float, ...]
    finishes: tuple[float, ...]
    completion_tokens: int

    @property
    def makespan(self) -> float:
        return max(self.finishes)

    @property
    def latencies(self) -> list[float]:
        return [finish - arrival for finish, arrival in zip(self.finishes, self.arrivals)]


def run(
    cases: Sequence[PromptCase],
    *,
    url: str,
    served_model: str,
    max_in_flight: int,
    warmups: int,
    repeats: int,
    decode_tokens: int,
    requests: int,
    interval_seconds: float,
    transport: Any = None,
) -> list[tuple[str, ...]]:
    """Replay evenly spaced arrivals against ``url`` and report medians."""

    if requests <= 0:
        raise ValueError(f"requests must be positive, got {requests}")
    if interval_seconds < 0:
        raise ValueError(f"arrival interval must be non-negative, got {interval_seconds}")
    if max_in_flight <= 0:
        raise ValueError(f"max_in_flight must be positive, got {max_in_flight}")
    try:
        import httpx
    except ImportError as exc:
        raise StaggeredHttpError("staggered-http requires httpx; run uv sync") from exc

    endpoint = url.rstrip("/") + "/chat/completions"
    arrivals = tuple(index * interval_seconds for index in range(requests))
    rows: list[tuple[str, ...]] = []
    for case in cases:
        def replay() -> _Replay:
            return asyncio.run(
                _replay(
                    httpx,
                    transport,
                    endpoint,
                    served_model,
                    case,
                    arrivals,
                    max_in_flight,
                    decode_tokens,
                )
            )

        for _ in range(warmups):
            replay()
        runs = [replay() for _ in range(repeats)]
        makespan = median(run.makespan for run in runs)
        throughput = median(run.completion_tokens / run.makespan for run in runs)

        def stat(fraction: float) -> str:
            seconds = median(percentile(run.latencies, fraction) for run in runs)
            return f"{seconds * 1_000:.1f} ms"

        rows.append(
            (
                url,
                str(requests),
                f"{interval_seconds * 1_000:.0f} ms",
                str(max_in_flight),
                str(case.actual_tokens),
                f"{makespan:.2f} s",
                f"{throughput:.1f}",
                stat(0.5),
                stat(0.95),
            )
        )
    return rows


async def _replay(
    httpx: Any,
    transport: Any,
    endpoint: str,
    served_model: str,
    case: PromptCase,
    arrivals: Sequence[float],
    max_in_flight: int,
    decode_tokens: int,
) -> _Replay:
    slots = asyncio.Semaphore(max_in_flight)
    body = {
        "model": served_model,
        "messages": [{"role": "user", "content": case.prompt}],
        "temperature": 0,
        "max_completion_tokens": decode_tokens,
    }
    finishes: list[float | None] = [None] * len(arrivals)
    completion_tokens = 0

    async with httpx.AsyncClient(
        timeout=None,
        limits=httpx.Limits(max_connections=max_in_flight),
        transport=transport,
    ) as client:
        started = time.perf_counter()

        async def send(request: int) -> None:
            nonlocal completion_tokens
            delay = started + arrivals[request] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            # The slot is released only after the whole body has been read,
            # so the client never has more requests in flight than the
            # server counts.
            async with slots:
                try:
                    response = await client.post(endpoint, json=body)
                except httpx.HTTPError as exc:
                    raise StaggeredHttpError(
                        f"request {request}: {endpoint}: {exc!r}"
                    ) from exc
                finished = time.perf_counter() - started
            if response.status_code != 200:
                raise StaggeredHttpError(
                    f"request {request}: HTTP {response.status_code}: {response.text}"
                )
            usage = response.json().get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens"):
                if not isinstance(usage.get(name), int):
                    raise StaggeredHttpError(
                        f"request {request}: response has no integer usage.{name}"
                    )
            if usage["prompt_tokens"] != case.actual_tokens:
                raise StaggeredHttpError(
                    f"request {request}: server counted {usage['prompt_tokens']} "
                    f"prompt tokens, expected {case.actual_tokens}"
                )
            completion_tokens += usage["completion_tokens"]
            finishes[request] = finished

        tasks = [asyncio.create_task(send(request)) for request in range(len(arrivals))]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.exception() is not None:
                raise task.exception()

    return _Replay(
        arrivals=tuple(arrivals),
        finishes=tuple(finishes),  # type: ignore[arg-type]
        completion_tokens=completion_tokens,
    )
