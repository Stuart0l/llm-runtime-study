"""Batched autoregressive generation using prefill and cached decoding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import time
from typing import Callable, Iterator, Literal, Sequence, TypeVar

import torch

from mini_llm.cache import KVCacheManager, SequenceKVCache
from mini_llm.model.contracts import RuntimeCausalLM
from mini_llm.sampling import (
    SamplingConfig,
    SamplingError,
    make_generator,
    sample_next_token,
)
from mini_llm.tokenizer import ChatMessage, RuntimeTokenizer


class GenerationError(ValueError):
    """Raised when a generation request cannot be executed."""


FinishReason = Literal["eos", "max_new_tokens", "context_length"]
DecodeFactory = Callable[
    [Sequence[SequenceKVCache]], Callable[[torch.Tensor], torch.Tensor]
]


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
class _WaitingRequest:
    index: int
    prompt_token_ids: list[int]
    output_limit: int
    max_new_tokens: int
    sampling: SamplingConfig


@dataclass(slots=True)
class _BatchRequest:
    index: int
    prompt_length: int
    output_limit: int
    max_new_tokens: int
    sampling: SamplingConfig
    cache: SequenceKVCache
    generator: torch.Generator | None
    text_decoder: "IncrementalTextDecoder"
    logits: torch.Tensor
    model_seconds: float | None
    token_index: int = 0
    next_token_id: int | None = None


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


class BatchScheduler:
    """Admit, prefill and decode requests as one continuously changing batch."""

    def __init__(
        self,
        model: RuntimeCausalLM,
        tokenizer: RuntimeTokenizer,
        *,
        cache_manager: KVCacheManager,
        max_batch_size: int,
        max_seq_len: int | None = None,
        synchronize: Callable[[], None] | None = None,
        decode_factory: DecodeFactory | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise GenerationError("max_batch_size must be positive")
        model_context_limit = model.config.max_position_embeddings
        context_limit = model_context_limit if max_seq_len is None else max_seq_len
        if context_limit <= 0 or context_limit > model_context_limit:
            raise GenerationError(
                f"max_seq_len must be within [1, {model_context_limit}], got "
                f"{context_limit}"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size
        self.context_limit = context_limit
        self._synchronize = synchronize
        self._decode_factory = decode_factory
        self._eos_token_ids = frozenset(model.config.eos_token_ids)
        self._request_ids = itertools.count()
        self._waiting: deque[_WaitingRequest] = deque()
        self._running: list[_BatchRequest] = []

    def add_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig = SamplingConfig(),
        enable_thinking: bool = False,
    ) -> int:
        """Validate and queue one request; return the index its events carry."""

        if max_new_tokens <= 0:
            raise GenerationError("max_new_tokens must be positive")
        vocab_size = self.model.config.vocab_size
        if sampling.top_k is not None and sampling.top_k > vocab_size:
            raise SamplingError(
                f"top_k {sampling.top_k} exceeds vocabulary size {vocab_size}"
            )
        formatted_prompt = self.tokenizer.format_chat(
            messages, enable_thinking=enable_thinking
        )
        prompt_token_ids = self.tokenizer.encode(formatted_prompt)
        prompt_length = len(prompt_token_ids)
        if prompt_length > self.context_limit:
            raise GenerationError(
                f"prompt has {prompt_length} tokens but model context "
                f"limit is {self.context_limit}"
            )
        output_limit = min(max_new_tokens, self.context_limit - prompt_length)
        required = prompt_length + output_limit
        if output_limit > 0 and required > self.cache_manager.capacity:
            raise GenerationError(
                f"request needs {required} cache tokens but the KV cache holds "
                f"{self.cache_manager.capacity}"
            )
        index = next(self._request_ids)
        self._waiting.append(
            _WaitingRequest(
                index=index,
                prompt_token_ids=prompt_token_ids,
                output_limit=output_limit,
                max_new_tokens=max_new_tokens,
                sampling=sampling,
            )
        )
        return index

    @property
    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def step(self) -> list[tuple[int, GenerationEvent]]:
        """Advance every request by one token and return the events produced.

        Events are ``(request_index, event)`` pairs using the index returned by
        :meth:`add_request`. Any failure aborts every request and re-raises.
        """

        events: list[tuple[int, GenerationEvent]] = []
        try:
            with torch.inference_mode():
                # 1. Feed the tokens sampled by the previous step through one
                #    batched forward pass. Requests admitted below skip this:
                #    their first logits come from their own prefill.
                self._decode()
                # 2. Fill free batch slots from the queue and prefill them, so
                #    new requests join before this step's sampling.
                self._admit(events)
            # 3. Sample last, outside inference mode: every running request,
            #    old or newly admitted, emits exactly one token event now. A
            #    request that finishes on this token is released here and never
            #    pays for another decode; the rest carry the token into the
            #    next step's decode.
            self._sample(events)
        except Exception:
            # A failed prefill, decode or sample leaves the batch in an unknown
            # state (caches may be half-extended), so no request can continue.
            # Free every cache before propagating so nothing leaks.
            self.abort_all()
            raise
        return events

    def abort_all(self) -> None:
        """Release every running cache and drop every waiting request."""

        running, self._running = self._running, []
        self._waiting.clear()
        for request in running:
            self.cache_manager.release(request.cache)

    def _decode(self) -> None:
        running = self._running
        if not running:
            return
        # Every running request holds exactly one pending token: the one
        # sampled at the end of the previous step. Row i of the batch is
        # running[i], and that order is kept for the logits below.
        token_inputs = torch.tensor(
            [request.next_token_id for request in running],
            dtype=torch.long,
            device=self.model.input_device,
        ).unsqueeze(1)
        caches = tuple(request.cache for request in running)
        if self._decode_factory is not None:
            # The factory binds this exact request set to a decode function
            # (a CUDA graph for this batch size, captured on first use). Call
            # it every step, since membership changes whenever a request joins
            # or finishes, and call it here rather than inside the timed
            # operation so capture and rebinding don't count as model time.
            decode = self._decode_factory(caches)
            operation = lambda: decode(token_inputs)
        else:
            operation = lambda: self.model.decode(token_inputs, caches=caches)
        logits, model_seconds = _run_model_call(operation, self._synchronize)
        for row, request in enumerate(running):
            # Keep a [1, 1, vocab] view per request so sampling reads the same
            # shape as prefill logits. The pending token is now in the cache.
            request.logits = logits[row : row + 1]
            request.model_seconds = model_seconds
            request.next_token_id = None

    def _admit(self, events: list[tuple[int, GenerationEvent]]) -> None:
        while self._waiting:
            waiting = self._waiting[0]
            prompt_length = len(waiting.prompt_token_ids)
            if waiting.output_limit == 0:
                # The prompt already fills the context limit, so no token can
                # be generated. Finish it without a slot, a cache or a forward
                # pass, and keep admitting behind it.
                self._waiting.popleft()
                events.append(
                    (
                        waiting.index,
                        GenerationEvent(
                            token_id=None,
                            token_index=None,
                            text_delta="",
                            text="",
                            finish_reason="context_length",
                            prompt_token_count=prompt_length,
                        ),
                    )
                )
                continue

            # Reserve the whole lifetime up front (prompt plus every output
            # token), so an admitted request can never run out of cache
            # mid-generation and nothing ever needs preemption.
            capacity = prompt_length + waiting.output_limit
            # Strict FCFS: if the head doesn't fit, stop instead of letting
            # smaller requests behind it jump ahead. add_request already
            # rejected requests larger than the whole cache, so the head
            # always fits eventually, once running requests release theirs.
            if len(self._running) >= self.max_batch_size:
                return
            if not self.cache_manager.can_allocate(capacity):
                return
            self._waiting.popleft()
            cache = self.cache_manager.allocate(capacity)
            # Prefill one request at a time (batch size one); running
            # requests wait for it. That delay is paid once per admission.
            prompt_tokens = torch.tensor(
                [waiting.prompt_token_ids],
                dtype=torch.long,
                device=self.model.input_device,
            )
            try:
                logits, model_seconds = _run_model_call(
                    lambda: self.model.prefill(prompt_tokens, cache=cache),
                    self._synchronize,
                )
            except Exception:
                # This cache isn't in _running yet, so abort_all in step()
                # would miss it. Release it here before propagating.
                self.cache_manager.release(cache)
                raise
            # The prefill logits predict the first output token; _sample
            # consumes them later in this same step.
            self._running.append(
                _BatchRequest(
                    index=waiting.index,
                    prompt_length=prompt_length,
                    output_limit=waiting.output_limit,
                    max_new_tokens=waiting.max_new_tokens,
                    sampling=waiting.sampling,
                    cache=cache,
                    generator=make_generator(waiting.sampling.seed),
                    text_decoder=IncrementalTextDecoder(self.tokenizer),
                    logits=logits,
                    model_seconds=model_seconds,
                )
            )

    def _sample(self, events: list[tuple[int, GenerationEvent]]) -> None:
        continuing = []
        for request in self._running:
            # Each request samples with its own settings and seeded generator,
            # so its output doesn't depend on which requests share the batch.
            token_id = sample_next_token(
                request.logits[0, -1],
                request.sampling,
                generator=request.generator,
            )
            text_delta = request.text_decoder.add(token_id)
            request.token_index += 1

            # EOS wins over the length limits. output_limit is
            # min(max_new_tokens, room left in the context), so reaching it
            # means whichever of the two was smaller.
            finish_reason: FinishReason | None = None
            if token_id in self._eos_token_ids:
                finish_reason = "eos"
            elif request.token_index == request.output_limit:
                finish_reason = (
                    "context_length"
                    if request.output_limit < request.max_new_tokens
                    else "max_new_tokens"
                )

            events.append(
                (
                    request.index,
                    GenerationEvent(
                        token_id=token_id,
                        token_index=request.token_index - 1,
                        text_delta=text_delta,
                        text=request.text_decoder.text,
                        finish_reason=finish_reason,
                        model_seconds=request.model_seconds,
                        # Report prompt size once, on the request's first token.
                        prompt_token_count=(
                            request.prompt_length
                            if request.token_index == 1
                            else None
                        ),
                    ),
                )
            )
            if finish_reason is None:
                # Not yet in the cache: the next step's _decode writes it.
                request.next_token_id = token_id
                continuing.append(request)
            else:
                # Free the blocks now so _admit in the next step can reuse them.
                self.cache_manager.release(request.cache)
        self._running = continuing


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
    decode_factory: DecodeFactory | None = None,
    max_batch_size: int | None = None,
) -> Iterator[tuple[int, GenerationEvent]]:
    """Generate one or more requests and identify their streamed events.

    Requests beyond ``max_batch_size`` (default: all of them) wait and join the
    running batch as earlier requests finish.
    """

    if not message_batches:
        raise GenerationError("a generation batch must contain at least one request")
    scheduler = BatchScheduler(
        model,
        tokenizer,
        cache_manager=cache_manager,
        max_batch_size=(
            len(message_batches) if max_batch_size is None else max_batch_size
        ),
        max_seq_len=max_seq_len,
        synchronize=synchronize,
        decode_factory=decode_factory,
    )
    for messages in message_batches:
        scheduler.add_request(
            messages,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            enable_thinking=enable_thinking,
        )

    def iterate() -> Iterator[tuple[int, GenerationEvent]]:
        try:
            while scheduler.has_unfinished:
                yield from scheduler.step()
        finally:
            scheduler.abort_all()

    return iterate()
