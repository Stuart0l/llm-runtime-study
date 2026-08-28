"""FastAPI application exposing synchronous OpenAI Chat Completions."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterator, Protocol, Sequence, TextIO

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from mini_llm.cache import KVCacheError
from mini_llm.checkpoint import CheckpointValidationError
from mini_llm.config import ConfigError
from mini_llm.engine import Engine, EngineError
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
    # Each loaded causal model owns one mutable KV cache. asyncio.Lock queues
    # concurrent requests before they can enter generation.
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


def _port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be within [1, 65535]")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mini_llm.server",
        description="Serve the local Qwen3 runtime through Chat Completions.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port_number, default=8000)
    parser.add_argument("--served-model-name")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    return parser


def run_server(args: argparse.Namespace, *, output: TextIO) -> None:
    """Load one engine, build one application, and start one Uvicorn worker."""

    served_model = args.served_model_name or args.model.name
    print(f"Loading {served_model} from {args.model}...", file=output, flush=True)
    engine = Engine.from_model_dir(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )
    app = create_app(engine, served_model=served_model)
    print(
        f"Loaded on {engine.device} with {engine.dtype} in "
        f"{engine.load_seconds:.2f} s.",
        file=output,
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    output: TextIO | None = None,
    error: TextIO | None = None,
) -> int:
    output = sys.stdout if output is None else output
    error = sys.stderr if error is None else error
    args = build_parser().parse_args(argv)
    try:
        run_server(args, output=output)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=output)
    except (
        CheckpointValidationError,
        ConfigError,
        EngineError,
        TokenizerError,
    ) as exc:
        print(f"error: {exc}", file=error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
