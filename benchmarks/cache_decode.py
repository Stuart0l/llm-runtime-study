"""Cached versus full-sequence decode benchmark."""

from __future__ import annotations

from typing import Sequence

import torch

from mini_llm.engine import Engine

from benchmarks.common import PromptCase, input_tensor, measure


HEADERS = (
    "device",
    "prompt",
    "cached TPOT",
    "cached tok/s",
    "uncached TPOT",
    "uncached tok/s",
    "cached speedup",
    "cache",
    "max error",
    "next equal",
)

# Full-sequence decode repeats prompt prefill for every generated token and
# becomes prohibitively slow at the larger default prompt lengths. The short
# case is sufficient to demonstrate the algorithmic cache speedup.
UNCACHED_MAX_REQUESTED_PROMPT_TOKENS = 32


def _forced_tokens(
    engine: Engine, prompt: torch.Tensor, decode_tokens: int
) -> list[int]:
    engine.model.setup_cache(prompt.shape[1] + decode_tokens)
    logits = engine.model.prefill(prompt)
    tokens: list[int] = []
    for _ in range(decode_tokens):
        token_id = int(logits[0, -1].argmax())
        tokens.append(token_id)
        logits = engine.model.decode(
            torch.tensor([[token_id]], dtype=torch.long, device=engine.device)
        )
    return tokens


def run(
    engine: Engine,
    cases: Sequence[PromptCase],
    *,
    warmups: int,
    repeats: int,
    decode_tokens: int,
) -> list[tuple[str, ...]]:
    """Benchmark both decode algorithms on an already-loaded engine."""

    if decode_tokens <= 0:
        raise ValueError(f"decode tokens must be positive, got {decode_tokens}")

    rows: list[tuple[str, ...]] = []
    with torch.inference_mode():
        for case in cases:
            if case.requested_tokens > UNCACHED_MAX_REQUESTED_PROMPT_TOKENS:
                continue

            prompt = input_tensor(engine, case)
            forced_tokens = _forced_tokens(engine, prompt, decode_tokens)
            token_inputs = [
                torch.tensor([[token]], dtype=torch.long, device=engine.device)
                for token in forced_tokens
            ]
            prefixes = [
                torch.tensor(
                    [case.token_ids + tuple(forced_tokens[: index + 1])],
                    dtype=torch.long,
                    device=engine.device,
                )
                for index in range(decode_tokens)
            ]
            engine.model.setup_cache(case.actual_tokens + decode_tokens)

            def prepare_cached() -> None:
                engine.model.prefill(prompt)

            def cached_decode() -> torch.Tensor:
                cached_logits = None
                for token_input in token_inputs:
                    cached_logits = engine.model.decode(token_input)
                assert cached_logits is not None
                # Decode returns only one position, so retaining the final
                # measured logits is cheap and avoids rerunning cached decode
                # later for the correctness comparison.
                return cached_logits

            def uncached_decode() -> torch.Tensor:
                uncached_logits = None
                for prefix in prefixes:
                    uncached_logits = engine.model(prefix)
                assert uncached_logits is not None
                return uncached_logits

            cached = measure(
                cached_decode,
                synchronize=engine.synchronize,
                warmups=warmups,
                repeats=repeats,
                prepare=prepare_cached,
                capture=lambda logits: logits[0, -1].float(),
            )
            cached_tpot = cached.median_seconds / decode_tokens
            cache = engine.model.cache
            assert cache is not None

            uncached = measure(
                uncached_decode,
                synchronize=engine.synchronize,
                warmups=warmups,
                repeats=repeats,
                capture=lambda logits: logits[0, -1].float(),
            )
            cached_logits = cached.results[-1]
            uncached_logits = uncached.results[-1]
            max_error = float((cached_logits - uncached_logits).abs().max())
            next_equal = int(cached_logits.argmax()) == int(
                uncached_logits.argmax()
            )
            uncached_tpot = uncached.median_seconds / decode_tokens

            rows.append(
                (
                    engine.device.type,
                    str(case.actual_tokens),
                    f"{cached_tpot * 1_000:.2f} ms",
                    f"{1 / cached_tpot:.2f}",
                    f"{uncached_tpot * 1_000:.2f} ms",
                    f"{1 / uncached_tpot:.2f}",
                    f"{uncached_tpot / cached_tpot:.2f}x",
                    f"{cache.num_bytes / (1024**2):.2f} MiB",
                    f"{max_error:.4f}",
                    "yes" if next_equal else "no",
                )
            )
    return rows
