"""Show the complete formatted history used for multi-turn generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_llm.engine import Engine
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    args = parser.parse_args()

    messages = [
        ChatMessage("system", "Answer briefly and clearly."),
        ChatMessage("user", "What does a KV cache store?"),
        ChatMessage("assistant", "It stores attention keys and values."),
        ChatMessage("user", "Why does that make decoding faster?"),
    ]
    engine = Engine.from_model_dir(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    print("Formatted prompt:\n")
    print(engine.tokenizer.format_chat(messages, enable_thinking=False))

    print("Generated continuation:\n")
    for event in engine.generate(
        messages,
        max_new_tokens=args.max_new_tokens,
        sampling=SamplingConfig(temperature=0),
    ):
        print(event.text_delta, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
