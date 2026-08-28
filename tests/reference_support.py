"""Shared execution helpers for optional Transformers model tests."""

from __future__ import annotations

import gc
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import torch

from mini_llm.interfaces import ChatMessage, RuntimeCausalLM, RuntimeTokenizer


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@dataclass(frozen=True, slots=True)
class InferenceTrace:
    """Logits and greedy tokens from one runtime's prefill/decode sequence."""

    full_logits: torch.Tensor
    prefill_logits: torch.Tensor
    decode_logits: tuple[torch.Tensor, ...]
    token_ids: tuple[int, ...]


def run_mini_runtime(
    model: RuntimeCausalLM,
    input_ids: torch.Tensor,
    *,
    generated_tokens: int,
) -> InferenceTrace:
    model.setup_cache(input_ids.shape[1] + generated_tokens)
    with torch.inference_mode():
        full_logits = model(input_ids).float().clone()
        prefill_logits = model.prefill(input_ids).float().clone()
        token_ids = [int(prefill_logits[0, -1].argmax())]
        decode_logits = []
        for _ in range(generated_tokens - 1):
            token_input = torch.tensor([[token_ids[-1]]], dtype=torch.long)
            logits = model.decode(token_input).float().clone()
            decode_logits.append(logits)
            token_ids.append(int(logits[0, -1].argmax()))
    return InferenceTrace(
        full_logits,
        prefill_logits,
        tuple(decode_logits),
        tuple(token_ids),
    )


def run_transformers(
    model_dir: Path,
    input_ids: torch.Tensor,
    *,
    generated_tokens: int,
) -> InferenceTrace:
    # Transformers is an optional correctness oracle, never a runtime dependency.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True)
        prefill_logits = prefill.logits.float().clone()
        token_ids = [int(prefill_logits[0, -1].argmax())]
        past_key_values = prefill.past_key_values
        decode_logits = []
        for _ in range(generated_tokens - 1):
            decode = model(
                input_ids=torch.tensor([[token_ids[-1]]], dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = decode.past_key_values
            logits = decode.logits.float().clone()
            decode_logits.append(logits)
            token_ids.append(int(logits[0, -1].argmax()))
    del model
    gc.collect()
    return InferenceTrace(
        prefill_logits,
        prefill_logits,
        tuple(decode_logits),
        tuple(token_ids),
    )


def formatted_input(
    tokenizer: RuntimeTokenizer,
    content: str,
) -> torch.Tensor:
    prompt = tokenizer.format_chat(
        [ChatMessage(role="user", content=content)],
        enable_thinking=False,
    )
    return torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
