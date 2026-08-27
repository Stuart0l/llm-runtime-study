"""Load the local Qwen3 checkpoint and run an uncached forward pass."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from mini_llm.model import Qwen3ForCausalLM
from mini_llm.tokenizer import Qwen3Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--text", default="Hello from the mini runtime")
    args = parser.parse_args()

    tokenizer = Qwen3Tokenizer.from_model_dir(args.model_dir)
    input_ids = torch.tensor([tokenizer.encode(args.text)], dtype=torch.long)

    load_started = time.perf_counter()
    model = Qwen3ForCausalLM.from_model_dir(args.model_dir)
    load_seconds = time.perf_counter() - load_started
    with torch.inference_mode():
        forward_started = time.perf_counter()
        logits = model(input_ids)
        forward_seconds = time.perf_counter() - forward_started

    next_token_id = int(logits[0, -1].argmax().item())
    print(f"input IDs:       {input_ids.tolist()[0]}")
    print(f"logits shape:    {tuple(logits.shape)}")
    print(f"logits dtype:    {logits.dtype}")
    print(f"finite logits:   {bool(torch.isfinite(logits).all())}")
    print(f"greedy next ID:  {next_token_id}")
    print(f"greedy next text:{tokenizer.decode([next_token_id])!r}")
    print(f"load time:       {load_seconds:.3f} s")
    print(f"forward time:    {forward_seconds:.3f} s")


if __name__ == "__main__":
    main()
