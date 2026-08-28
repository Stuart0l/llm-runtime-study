"""Compare cached and uncached logits for a supported model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from mini_llm.config import load_config
from mini_llm.interfaces import ChatMessage
from mini_llm.model_loader import load_model
from mini_llm.tokenizer import load_tokenizer


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading the wall clock."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="Explain grouped-query attention briefly.")
    parser.add_argument("--decode-steps", type=int, default=3)
    args = parser.parse_args()
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be positive")

    config = load_config(args.model_dir)
    tokenizer = load_tokenizer(args.model_dir, model_config=config)
    prompt = tokenizer.format_chat(
        [ChatMessage(role="user", content=args.prompt)], enable_thinking=False
    )
    prompt_ids = tokenizer.encode(prompt)
    model = load_model(args.model_dir, model_config=config)
    model.setup_cache(len(prompt_ids) + args.decode_steps)
    full_ids = torch.tensor([prompt_ids], dtype=torch.long)
    device = full_ids.device

    with torch.inference_mode():
        uncached = model(full_ids)
        cached = model.prefill(full_ids)
        prefill_error = (cached - uncached).abs().max().item()
        print(
            f"prefill: tokens={len(prompt_ids)}, cache_length={model.cache.length}, "
            f"max_logit_error={prefill_error:.8f}"
        )

        generated: list[int] = []
        cached_decode_seconds = 0.0
        uncached_decode_seconds = 0.0
        next_token = int(cached[0, -1].argmax().item())
        for step in range(args.decode_steps):
            generated.append(next_token)
            token_input = torch.tensor([[next_token]], dtype=torch.long)
            full_ids = torch.cat((full_ids, token_input), dim=1)

            synchronize(device)
            started = time.perf_counter()
            cached = model.decode(token_input)
            synchronize(device)
            cached_decode_seconds += time.perf_counter() - started

            synchronize(device)
            started = time.perf_counter()
            uncached = model(full_ids)
            synchronize(device)
            uncached_decode_seconds += time.perf_counter() - started

            error = (cached[0, -1] - uncached[0, -1]).abs().max().item()
            cached_next = int(cached[0, -1].argmax().item())
            uncached_next = int(uncached[0, -1].argmax().item())
            print(
                f"decode {step + 1}: cache_length={model.cache.length}, "
                f"max_logit_error={error:.8f}, "
                f"next_token_equal={cached_next == uncached_next}"
            )
            next_token = cached_next

    cached_tpot = cached_decode_seconds / args.decode_steps
    uncached_tpot = uncached_decode_seconds / args.decode_steps
    print(f"generated text: {tokenizer.decode(generated)!r}")
    print(f"cache bytes: {model.cache.num_bytes:,}")
    print("\ndecode timing (prefill excluded)")
    print(
        f"  cached:   TPOT={cached_tpot * 1_000:.2f} ms, "
        f"throughput={1 / cached_tpot:.2f} tokens/s"
    )
    print(
        f"  uncached: TPOT={uncached_tpot * 1_000:.2f} ms, "
        f"throughput={1 / uncached_tpot:.2f} tokens/s"
    )
    print(f"  cached speedup: {uncached_tpot / cached_tpot:.2f}x")


if __name__ == "__main__":
    main()
