"""Shared benchmark inputs, synchronized timing, and table rendering."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import time
from typing import Callable, Generic, Sequence, TypeVar

import torch

from mini_llm.engine import Engine
from mini_llm.interfaces import ChatMessage


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One deterministic raw prompt and its formatted token IDs."""

    requested_tokens: int
    prompt: str
    token_ids: tuple[int, ...]

    @property
    def actual_tokens(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class Measurements(Generic[T]):
    """Measured durations and corresponding operation results."""

    seconds: tuple[float, ...]
    results: tuple[T, ...]

    @property
    def median_seconds(self) -> float:
        return median(self.seconds)


_PROMPT_FRAGMENT = (
    "Explain how a compact transformer inference runtime processes this prompt. "
)


def _formatted_ids(engine: Engine, prompt: str) -> tuple[int, ...]:
    formatted = engine.tokenizer.format_chat(
        [ChatMessage("user", prompt)], enable_thinking=False
    )
    return tuple(engine.tokenizer.encode(formatted))


def build_prompt_case(engine: Engine, target_tokens: int) -> PromptCase:
    """Find a deterministic natural-language prompt close to a token target."""

    if target_tokens <= 0:
        raise ValueError(f"prompt length must be positive, got {target_tokens}")

    # Tokenization is not linear in character count, so find the closest of two
    # neighboring repetition counts rather than claiming an exact token length.
    low = 0
    high = 1
    while len(_formatted_ids(engine, _PROMPT_FRAGMENT * high)) < target_tokens:
        low = high
        high *= 2

    while low + 1 < high:
        middle = (low + high) // 2
        if len(_formatted_ids(engine, _PROMPT_FRAGMENT * middle)) < target_tokens:
            low = middle
        else:
            high = middle

    candidates = []
    for repetitions in sorted({low, high}):
        prompt = _PROMPT_FRAGMENT * repetitions
        token_ids = _formatted_ids(engine, prompt)
        candidates.append(
            PromptCase(target_tokens, prompt, token_ids)
        )
    return min(
        candidates,
        key=lambda case: (abs(case.actual_tokens - target_tokens), case.actual_tokens),
    )


def measure(
    operation: Callable[[], T],
    *,
    synchronize: Callable[[], None],
    warmups: int,
    repeats: int,
    prepare: Callable[[], None] | None = None,
    capture: Callable[[T], T] | None = None,
) -> Measurements[T]:
    """Run warmups and capture compact results outside the timed window."""

    if warmups < 0:
        raise ValueError(f"warmups must be non-negative, got {warmups}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    # Warmups exercise lazy kernel setup, allocator growth, and reusable KV
    # cache allocation. Synchronization finishes asynchronous MPS work, but no
    # warmup wall time or result is recorded.
    for _ in range(warmups):
        if prepare is not None:
            prepare()
        synchronize()
        operation()
        synchronize()

    seconds: list[float] = []
    results: list[T] = []
    for _ in range(repeats):
        if prepare is not None:
            prepare()
        synchronize()
        started = time.perf_counter()
        result = operation()
        synchronize()
        seconds.append(time.perf_counter() - started)
        # A model prefill can return [batch, sequence, vocabulary], which is
        # expensive to retain across repetitions. Reduce it after timing so
        # result collection does not affect the reported operation latency.
        results.append(result if capture is None else capture(result))
    return Measurements(tuple(seconds), tuple(results))


def render_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a dependency-free, left-aligned terminal table."""

    if not headers:
        raise ValueError("table must have at least one column")
    rendered_rows = [[str(value) for value in row] for row in rows]
    if any(len(row) != len(headers) for row in rendered_rows):
        raise ValueError("every table row must match the header width")
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rendered_rows)
    return "\n".join(lines)


def input_tensor(engine: Engine, case: PromptCase) -> torch.Tensor:
    """Place one prompt's token IDs on the engine input device."""

    return torch.tensor(
        [case.token_ids], dtype=torch.long, device=engine.device
    )
