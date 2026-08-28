"""Compare sampling strategies on one supported model."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_llm.config import load_config
from mini_llm.generation import generate
from mini_llm.model_loader import load_model
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Explain KV caching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir, model_config=config)
    model = load_model(args.model_dir, model_config=config)

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
