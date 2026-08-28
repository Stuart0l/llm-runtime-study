"""FastAPI application exposing synchronous OpenAI Chat Completions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mini_llm.cache import KVCacheError
from mini_llm.generation import FinishReason, GenerationError, GenerationEvent
from mini_llm.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIRequestError,
    create_chat_completion_response,
    prepare_chat_completion_request,
    validation_errors_response,
)
from mini_llm.sampling import SamplingConfig, SamplingError
from mini_llm.tokenizer import ChatMessage, TokenizerError


class GenerationEngine(Protocol):
    """The engine behavior needed by the HTTP adapter."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
        enable_thinking: bool = False,
    ) -> Iterator[GenerationEvent]: ...


@dataclass(frozen=True, slots=True)
class _CompletedGeneration:
    text: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


def _consume_generation(events: Iterator[GenerationEvent]) -> _CompletedGeneration:
    """Consume the runtime's token events into one HTTP response value."""

    final_event: GenerationEvent | None = None
    prompt_tokens: int | None = None
    completion_tokens = 0
    for event in events:
        final_event = event
        if event.prompt_token_count is not None:
            prompt_tokens = event.prompt_token_count
        if event.token_id is not None:
            completion_tokens += 1

    if final_event is None or final_event.finish_reason is None:
        raise RuntimeError("generation ended without a terminal event")
    if prompt_tokens is None:
        raise RuntimeError("generation did not report its prompt token count")
    return _CompletedGeneration(
        text=final_event.text,
        finish_reason=final_event.finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def create_app(engine: GenerationEngine, *, served_model: str) -> FastAPI:
    """Create one API application around an already-loaded engine."""

    if not served_model:
        raise ValueError("served_model must not be empty")
    app = FastAPI(title="mini-llm", version="0.1.0")
    # Qwen3ForCausalLM owns one mutable KV cache. asyncio.Lock queues concurrent
    # valid requests in acquisition order, and to_thread keeps the event loop
    # responsive while the synchronous PyTorch generation call is running.
    generation_lock = asyncio.Lock()
    app.state.engine = engine
    app.state.served_model = served_model
    app.state.generation_lock = generation_lock

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        body = validation_errors_response(error.errors())
        return JSONResponse(status_code=400, content=body.model_dump(mode="json"))

    @app.exception_handler(OpenAIRequestError)
    async def openai_request_error(
        _request: Request, error: OpenAIRequestError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content=error.response.model_dump(mode="json"),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        prepared = prepare_chat_completion_request(
            request, served_model=served_model
        )
        try:
            async with generation_lock:
                completed = await asyncio.to_thread(
                    _generate_completion,
                    engine,
                    prepared.messages,
                    prepared.max_new_tokens,
                    prepared.sampling,
                )
        except (GenerationError, KVCacheError, SamplingError, TokenizerError) as error:
            raise OpenAIRequestError(str(error)) from error

        return create_chat_completion_response(
            model=served_model,
            text=completed.text,
            finish_reason=completed.finish_reason,
            prompt_tokens=completed.prompt_tokens,
            completion_tokens=completed.completion_tokens,
        )

    return app


def _generate_completion(
    engine: GenerationEngine,
    messages: list[ChatMessage],
    max_new_tokens: int,
    sampling: SamplingConfig,
) -> _CompletedGeneration:
    """Run one complete request inside the application's cache lock."""

    events = engine.generate(
        messages,
        max_new_tokens=max_new_tokens,
        sampling=sampling,
        enable_thinking=False,
    )
    return _consume_generation(events)
