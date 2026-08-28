"""Autoregressive generation using one prefill followed by cached decoding."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator, Literal, Sequence, TypeVar

import torch

from mini_llm.interfaces import ChatMessage, RuntimeCausalLM, RuntimeTokenizer
from mini_llm.sampling import SamplingConfig, make_generator, sample_next_token


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


class IncrementalTextDecoder:
    """Decode only tokens not already emitted as stable Unicode text."""

    def __init__(self, tokenizer: RuntimeTokenizer) -> None:
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


def generate(
    model: RuntimeCausalLM,
    tokenizer: RuntimeTokenizer,
    messages: Sequence[ChatMessage],
    *,
    max_new_tokens: int,
    sampling: SamplingConfig = SamplingConfig(),
    enable_thinking: bool = False,
    max_seq_len: int | None = None,
    synchronize: Callable[[], None] | None = None,
) -> Iterator[GenerationEvent]:
    """Format complete chat history and return its generation iterator.

    Formatting, tokenization, and request validation happen before this
    function returns. This lets an HTTP adapter reject an invalid request
    before it commits streaming response headers. Model execution remains lazy
    and begins when the returned iterator is consumed.
    """

    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")
    formatted_prompt = tokenizer.format_chat(
        messages, enable_thinking=enable_thinking
    )
    prompt_token_ids = tokenizer.encode(formatted_prompt)

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

    def iterate() -> Iterator[GenerationEvent]:
        # A local generator keeps model execution lazy for streaming, while all
        # request validation above has already happened eagerly.
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

        device = model.input_device
        prompt_tokens = torch.tensor(
            [prompt_token_ids], dtype=torch.long, device=device
        )
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

    return iterate()
