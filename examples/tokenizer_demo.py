"""Demonstrate architecture-specific chat formatting and tokenization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mini_llm.tokenizer import ChatMessage, TextTokenizer, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Format and tokenize a supported model's chat prompt."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--system")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="allow Qwen3 reasoning; Granite does not support this option",
    )
    return parser


def render_token_mapping(tokenizer: TextTokenizer, token_ids: Sequence[int]) -> str:
    """Render the exact vocabulary token associated with each encoded ID."""

    rows = [
        "Index  Token                       ID",
        "-----  --------------------------  ------",
    ]
    for index, token_id in enumerate(token_ids):
        token = repr(tokenizer.id_to_token(token_id))
        rows.append(f"{index:>5}  {token:<26}  {token_id:>6}")
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.model_dir)
    messages = []
    if args.system is not None:
        messages.append(ChatMessage("system", args.system))
    messages.append(ChatMessage("user", args.prompt))
    formatted = tokenizer.format_chat(messages, enable_thinking=args.thinking)
    token_ids = tokenizer.encode(formatted)

    print("Tokenizer summary:")
    print(f"  base BPE entries:       {tokenizer.base_vocab_size:,}")
    print(f"  decodable entries:      {tokenizer.vocab_size:,}")
    print(f"  model output rows:      {tokenizer.model_vocab_size:,}")
    print(f"  special tokens:         {tokenizer.special_tokens}")
    print("Formatted prompt:")
    print(repr(formatted))
    print(f"Token-to-ID mapping ({len(token_ids)} tokens):")
    print(render_token_mapping(tokenizer, token_ids))
    print("Decoded round trip:")
    print(repr(tokenizer.decode(token_ids)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
