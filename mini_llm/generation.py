"""Autoregressive generation using one prefill followed by cached decoding."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator, Literal, TypeVar

import torch

from mini_llm.model import Qwen3ForCausalLM
from mini_llm.sampling import SamplingConfig, make_generator, sample_next_token
from mini_llm.tokenizer import ChatMessage, Qwen3Tokenizer, format_qwen3_chat


class GenerationError(ValueError):
    """Raised when a generation request cannot be executed."""


FinishReason = Literal["eos", "max_new_tokens", "context_length"]


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """One streamed token, plus the authoritative decoded text so far."""

    token_id: int | None
    token_index: int | None
    text_delta: str
    text: str
    finish_reason: FinishReason | None = None
    model_seconds: float | None = None
    prompt_token_count: int | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class IncrementalTextDecoder:
    """Decode only tokens not already emitted as stable Unicode text."""

    def __init__(self, tokenizer: Qwen3Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pending_token_ids: list[int] = []
        self.text = ""

    def add(self, token_id: int) -> str:
        """Return newly completed text, buffering an incomplete byte sequence."""

        self.pending_token_ids.append(token_id)
        pending_text = self.tokenizer.decode(
            self.pending_token_ids, skip_special_tokens=True
        )
        if pending_text.endswith("\ufffd"):
            return ""

        self.pending_token_ids.clear()
        self.text += pending_text
        return pending_text


T = TypeVar("T")


def _run_model_call(
    operation: Callable[[], T], synchronize: Callable[[], None] | None
) -> tuple[T, float | None]:
    """Run and optionally time one synchronized prefill or decode call."""

    if synchronize is None:
        return operation(), None
    synchronize()
    started = time.perf_counter()
    result = operation()
    synchronize()
    return result, time.perf_counter() - started


def encode_prompt(
    tokenizer: Qwen3Tokenizer, prompt: str, *, enable_thinking: bool = False
) -> list[int]:
    """Apply Qwen3's chat protocol and tokenize one raw user prompt."""

    if not isinstance(prompt, str) or not prompt:
        raise GenerationError("prompt must be a non-empty string")
    formatted_prompt = format_qwen3_chat(
        [ChatMessage(role="user", content=prompt)],
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(formatted_prompt)


def generate(
    model: Qwen3ForCausalLM,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    sampling: SamplingConfig = SamplingConfig(),
    enable_thinking: bool = False,
    max_seq_len: int | None = None,
    synchronize: Callable[[], None] | None = None,
) -> Iterator[GenerationEvent]:
    """Format a raw user prompt and stream one cache-backed response.

    The prompt is evaluated exactly once. Each non-final generated token is
    then passed to ``decode`` to obtain logits for the following token.
    """

    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")
    prompt_token_ids = encode_prompt(
        tokenizer, prompt, enable_thinking=enable_thinking
    )

    prompt_length = len(prompt_token_ids)
    model_context_limit = model.config.max_position_embeddings
    context_limit = model_context_limit if max_seq_len is None else max_seq_len
    if context_limit <= 0 or context_limit > model_context_limit:
        raise GenerationError(
            f"max_seq_len must be within [1, {model_context_limit}], got "
            f"{context_limit}"
        )
    if prompt_length > context_limit:
        raise GenerationError(
            f"prompt has {prompt_length} tokens but model context limit is "
            f"{context_limit}"
        )

    available_positions = context_limit - prompt_length
    output_limit = min(max_new_tokens, available_positions)
    if output_limit == 0:
        yield GenerationEvent(
            token_id=None,
            token_index=None,
            text_delta="",
            text="",
            finish_reason="context_length",
            prompt_token_count=prompt_length,
        )
        return

    device = model.model.embed_tokens.weight.device
    prompt_tokens = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    model.setup_cache(prompt_length + output_limit)
    random_generator = make_generator(sampling.seed)
    eos_token_ids = set(model.config.eos_token_ids)
    text_decoder = IncrementalTextDecoder(tokenizer)

    with torch.inference_mode():
        logits, model_seconds = _run_model_call(
            lambda: model.prefill(prompt_tokens), synchronize
        )
    for token_index in range(output_limit):
        token_id = sample_next_token(
            logits[0, -1], sampling, generator=random_generator
        )
        text_delta = text_decoder.add(token_id)

        finish_reason: FinishReason | None = None
        if token_id in eos_token_ids:
            finish_reason = "eos"
        elif token_index + 1 == output_limit:
            finish_reason = (
                "context_length"
                if output_limit < max_new_tokens
                else "max_new_tokens"
            )

        yield GenerationEvent(
            token_id=token_id,
            token_index=token_index,
            text_delta=text_delta,
            text=text_decoder.text,
            finish_reason=finish_reason,
            model_seconds=model_seconds,
            prompt_token_count=prompt_length if token_index == 0 else None,
        )
        if finish_reason is not None:
            return

        token_input = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits, model_seconds = _run_model_call(
                lambda: model.decode(token_input), synchronize
            )
