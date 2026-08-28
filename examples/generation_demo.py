"""Compare greedy and seeded sampling strategies on one Qwen3 prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_llm.generation import generate
from mini_llm.model import Qwen3ForCausalLM
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage, Qwen3Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Explain KV caching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer.from_model_dir(args.model_dir)
    model = Qwen3ForCausalLM.from_model_dir(args.model_dir)

    strategies = {
        "greedy": SamplingConfig(temperature=0),
        "seeded": SamplingConfig(temperature=0.8, seed=42),
        "top-k": SamplingConfig(temperature=0.8, top_k=20, seed=42),
        "top-p": SamplingConfig(temperature=0.8, top_p=0.9, seed=42),
    }
    for name, sampling in strategies.items():
        events = list(
            generate(
                model,
                tokenizer,
                [ChatMessage(role="user", content=args.prompt)],
                max_new_tokens=args.max_new_tokens,
                sampling=sampling,
            )
        )
        final = events[-1]
        print(f"{name:>7}: text={final.text!r} stop={final.finish_reason}")


if __name__ == "__main__":
    main()
