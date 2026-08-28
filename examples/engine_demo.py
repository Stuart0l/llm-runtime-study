"""Compare short CPU and MPS generation with synchronized timing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import torch

from mini_llm.engine import Engine
from mini_llm.sampling import SamplingConfig


@dataclass(frozen=True, slots=True)
class Result:
    text: str
    generated_token_count: int
    time_to_first_token: float
    decode_seconds: float


def run(engine: Engine, prompt: str, max_new_tokens: int) -> Result:
    stream = engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        sampling=SamplingConfig(temperature=0),
    )
    started = time.perf_counter()
    first = next(stream)
    first_finished = time.perf_counter()
    events = [first]
    events.extend(stream)
    finished = time.perf_counter()
    return Result(
        text=events[-1].text,
        generated_token_count=sum(event.token_id is not None for event in events),
        time_to_first_token=first_finished - started,
        decode_seconds=finished - first_finished,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Explain grouped-query attention briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    args = parser.parse_args()

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    else:
        print("MPS is unavailable in this environment; running CPU only.")

    results: dict[str, Result] = {}
    for device in devices:
        engine = Engine.from_model_dir(
            args.model_dir,
            device=device,
            dtype="float16",
            max_seq_len=4096,
        )
        result = run(engine, args.prompt, args.max_new_tokens)
        results[device] = result
        decode_count = max(0, result.generated_token_count - 1)
        decode_rate = (
            decode_count / result.decode_seconds if decode_count else 0.0
        )
        print(
            f"{device}: dtype={engine.dtype}, load={engine.load_seconds:.2f}s, "
            f"TTFT={result.time_to_first_token * 1_000:.2f}ms, "
            f"decode={decode_rate:.2f} tokens/s"
        )
        print(f"     text={result.text!r}")

    if "mps" in results:
        print(f"CPU and MPS text equal: {results['cpu'].text == results['mps'].text}")


if __name__ == "__main__":
    main()
