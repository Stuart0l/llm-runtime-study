"""Batched autoregressive generation using prefill and cached decoding."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterator, Literal, Sequence, TypeVar

import torch

from mini_llm.cache import KVCacheManager, SequenceKVCache
from mini_llm.model.contracts import RuntimeCausalLM
from mini_llm.sampling import SamplingConfig, make_generator, sample_next_token
from mini_llm.tokenizer import ChatMessage, RuntimeTokenizer


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


@dataclass(slots=True)
class _BatchRequest:
    index: int
    prompt_length: int
    output_limit: int
    cache: SequenceKVCache
    generator: torch.Generator | None
    text_decoder: "IncrementalTextDecoder"
    logits: torch.Tensor
    model_seconds: float | None
    token_index: int = 0


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
    message_batches: Sequence[Sequence[ChatMessage]],
    *,
    max_new_tokens: int,
    sampling: SamplingConfig = SamplingConfig(),
    enable_thinking: bool = False,
    max_seq_len: int | None = None,
    synchronize: Callable[[], None] | None = None,
    cache_manager: KVCacheManager,
    decode_factory: (
        Callable[[SequenceKVCache], Callable[[torch.Tensor], torch.Tensor]] | None
    ) = None,
) -> Iterator[tuple[int, GenerationEvent]]:
    """Generate one or more requests and identify their streamed events."""

    if not message_batches:
        raise GenerationError("a generation batch must contain at least one request")
    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")

    model_context_limit = model.config.max_position_embeddings
    context_limit = model_context_limit if max_seq_len is None else max_seq_len
    if context_limit <= 0 or context_limit > model_context_limit:
        raise GenerationError(
            f"max_seq_len must be within [1, {model_context_limit}], got "
            f"{context_limit}"
        )

    prompts = []
    for messages in message_batches:
        formatted_prompt = tokenizer.format_chat(
            messages, enable_thinking=enable_thinking
        )
        prompt_token_ids = tokenizer.encode(formatted_prompt)
        if len(prompt_token_ids) > context_limit:
            raise GenerationError(
                f"prompt has {len(prompt_token_ids)} tokens but model context "
                f"limit is {context_limit}"
            )
        prompts.append(prompt_token_ids)

    def iterate() -> Iterator[tuple[int, GenerationEvent]]:
        active: list[_BatchRequest] = []
        single_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
        try:
            for index, prompt_token_ids in enumerate(prompts):
                prompt_length = len(prompt_token_ids)
                output_limit = min(
                    max_new_tokens, context_limit - prompt_length
                )
                if output_limit == 0:
                    yield index, GenerationEvent(
                        token_id=None,
                        token_index=None,
                        text_delta="",
                        text="",
                        finish_reason="context_length",
                        prompt_token_count=prompt_length,
                    )
                    continue

                cache = cache_manager.allocate(prompt_length + output_limit)
                prompt_tokens = torch.tensor(
                    [prompt_token_ids], dtype=torch.long, device=model.input_device
                )
                try:
                    with torch.inference_mode():
                        logits, model_seconds = _run_model_call(
                            lambda: model.prefill(prompt_tokens, cache=cache),
                            synchronize,
                        )
                except Exception:
                    cache_manager.release(cache)
                    raise
                active.append(
                    _BatchRequest(
                        index=index,
                        prompt_length=prompt_length,
                        output_limit=output_limit,
                        cache=cache,
                        generator=make_generator(sampling.seed),
                        text_decoder=IncrementalTextDecoder(tokenizer),
                        logits=logits,
                        model_seconds=model_seconds,
                    )
                )

            eos_token_ids = set(model.config.eos_token_ids)
            while active:
                continuing = []
                token_ids = []
                for request in active:
                    token_id = sample_next_token(
                        request.logits[0, -1],
                        sampling,
                        generator=request.generator,
                    )
                    text_delta = request.text_decoder.add(token_id)
                    request.token_index += 1

                    finish_reason: FinishReason | None = None
                    if token_id in eos_token_ids:
                        finish_reason = "eos"
                    elif request.token_index == request.output_limit:
                        finish_reason = (
                            "context_length"
                            if request.output_limit < max_new_tokens
                            else "max_new_tokens"
                        )

                    yield request.index, GenerationEvent(
                        token_id=token_id,
                        token_index=request.token_index - 1,
                        text_delta=text_delta,
                        text=request.text_decoder.text,
                        finish_reason=finish_reason,
                        model_seconds=request.model_seconds,
                        prompt_token_count=(
                            request.prompt_length
                            if request.token_index == 1
                            else None
                        ),
                    )
                    if finish_reason is None:
                        continuing.append(request)
                        token_ids.append(token_id)
                    else:
                        cache_manager.release(request.cache)

                active = continuing
                if not active:
                    break

                token_inputs = torch.tensor(
                    token_ids, dtype=torch.long, device=model.input_device
                ).unsqueeze(1)
                with torch.inference_mode():
                    if len(active) == 1 and decode_factory is not None:
                        if single_decode is None:
                            single_decode = decode_factory(active[0].cache)
                        operation = lambda: single_decode(token_inputs)
                    else:
                        operation = lambda: model.decode(
                            token_inputs,
                            caches=tuple(request.cache for request in active),
                        )
                    logits, model_seconds = _run_model_call(operation, synchronize)
                for row, request in enumerate(active):
                    request.logits = logits[row : row + 1]
                    request.model_seconds = model_seconds
        finally:
            for request in active:
                cache_manager.release(request.cache)

    return iterate()
