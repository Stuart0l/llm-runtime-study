"""Full-model Granite MoE prefill benchmark."""

from __future__ import annotations

from typing import Sequence

import torch

from mini_llm.engine import Engine

from benchmarks.common import PromptCase, input_tensor, measure


HEADERS = (
    "device",
    "method",
    "prompt",
    "latency",
    "tokens/s",
    "CPU max error",
    "next equal",
)


def run(
    engine: Engine,
    cases: Sequence[PromptCase],
    *,
    warmups: int,
    repeats: int,
    cpu_references: dict[int, torch.Tensor],
) -> list[tuple[str, ...]]:
    """Benchmark the automatic full-model MoE prefill implementation."""

    if engine.model.config.model_type != "granitemoe":
        raise ValueError("moe-prefill requires a Granite MoE model")

    method = "expert-loop" if engine.device.type == "cpu" else "padded-batch"
    rows: list[tuple[str, ...]] = []
    with torch.inference_mode():
        for case in cases:
            prompt = input_tensor(engine, case)
            engine.model.setup_cache(case.actual_tokens)

            def prefill() -> torch.Tensor:
                return engine.model.prefill(prompt)

            timings = measure(
                prefill,
                synchronize=engine.synchronize,
                warmups=warmups,
                repeats=repeats,
                capture=lambda logits: logits[0, -1].float().cpu(),
            )
            logits = timings.results[-1]

            if engine.device.type == "cpu":
                cpu_references[case.requested_tokens] = logits
                max_error = "reference"
                next_equal = "reference"
            else:
                reference = cpu_references.get(case.requested_tokens)
                if reference is None:
                    max_error = "n/a"
                    next_equal = "n/a"
                else:
                    max_error = f"{float((logits - reference).abs().max()):.4f}"
                    next_equal = (
                        "yes"
                        if int(logits.argmax()) == int(reference.argmax())
                        else "no"
                    )

            seconds = timings.median_seconds
            rows.append(
                (
                    engine.device.type,
                    method,
                    str(case.actual_tokens),
                    f"{seconds * 1_000:.2f} ms",
                    f"{case.actual_tokens / seconds:.2f}",
                    max_error,
                    next_equal,
                )
            )
    return rows
