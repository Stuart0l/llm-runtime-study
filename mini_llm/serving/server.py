"""FastAPI application exposing synchronous OpenAI Chat Completions."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
import sys
import traceback
from typing import Protocol, Sequence, TextIO

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import torch
import uvicorn

from mini_llm.cache import KVCacheError
from mini_llm.checkpoint import CheckpointValidationError
from mini_llm.config import ConfigError
from mini_llm.engine import Engine, EngineError
from mini_llm.generation import FinishReason, GenerationError, GenerationEvent
from mini_llm.serving.openai_api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIRequestError,
    create_chat_completion_response,
    prepare_chat_completion_request,
    validation_errors_response,
)
from mini_llm.sampling import SamplingConfig, SamplingError
from mini_llm.tokenizer import ChatMessage, TokenizerError

# Errors that mean the request itself is invalid. Anything else raised while
# generating is a server-side failure and becomes HTTP 500.
_REQUEST_ERRORS = (GenerationError, KVCacheError, SamplingError, TokenizerError)


class ServingScheduler(Protocol):
    """The continuous-batching scheduler behaviour the HTTP adapter needs."""

    max_batch_size: int

    @property
    def has_unfinished(self) -> bool: ...

    def add_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
        enable_thinking: bool,
    ) -> int: ...

    def step(self) -> list[tuple[int, GenerationEvent]]: ...

    def abort_all(self) -> None: ...


class ServingEngine(Protocol):
    """The engine behaviour the HTTP adapter needs."""

    def create_scheduler(self) -> ServingScheduler: ...


@dataclass(frozen=True, slots=True)
class _CompletedGeneration:
    text: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class _RequestResult:
    """Accumulate one request's token events into its HTTP response value."""

    text: str = ""
    finish_reason: FinishReason | None = None
    prompt_tokens: int | None = None
    completion_tokens: int = 0

    def add(self, event: GenerationEvent) -> None:
        self.text = event.text
        if event.prompt_token_count is not None:
            self.prompt_tokens = event.prompt_token_count
        if event.token_id is not None:
            self.completion_tokens += 1
        if event.finish_reason is not None:
            self.finish_reason = event.finish_reason

    def completed(self) -> _CompletedGeneration:
        if self.finish_reason is None:
            raise RuntimeError("generation ended without a terminal event")
        if self.prompt_tokens is None:
            raise RuntimeError("generation did not report its prompt token count")
        return _CompletedGeneration(
            text=self.text,
            finish_reason=self.finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


@dataclass(slots=True)
class _Submitted:
    messages: list[ChatMessage]
    max_new_tokens: int
    sampling: SamplingConfig
    future: asyncio.Future[_CompletedGeneration]


@dataclass(slots=True)
class _IterationOutcome:
    added: list[int | Exception] = field(default_factory=list)
    events: list[tuple[int, GenerationEvent]] = field(default_factory=list)
    step_error: Exception | None = None


def _generation_failed(error: BaseException) -> OpenAIRequestError:
    return OpenAIRequestError(
        str(error) or type(error).__name__,
        status_code=500,
        error_type="server_error",
        code="generation_failed",
    )


class _SchedulerDriver:
    """Own one scheduler; run every scheduler call on one dedicated thread.

    All bookkeeping below is read and written on the event-loop thread only.
    Every call into the scheduler (``add_request``, ``has_unfinished``,
    ``step``, ``abort_all``) runs on the single-worker executor, which runs
    one job at a time, so those calls never overlap and the tokenizer is never
    used from two threads at once.
    """

    def __init__(self, engine: ServingEngine) -> None:
        # Creating the scheduler also creates the engine's KV cache manager,
        # which allocates the paged KV pool.
        self.scheduler = engine.create_scheduler()
        self.max_batch_size = self.scheduler.max_batch_size
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mini-llm-scheduler"
        )
        self.submitted: list[_Submitted] = []
        self.handing_over: list[_Submitted] = []
        self.in_scheduler: dict[
            int, tuple[asyncio.Future[_CompletedGeneration], _RequestResult]
        ] = {}
        self.wake = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    @property
    def in_flight(self) -> int:
        return len(self.submitted) + len(self.handing_over) + len(self.in_scheduler)

    def submit(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        sampling: SamplingConfig,
    ) -> asyncio.Future[_CompletedGeneration]:
        future: asyncio.Future[_CompletedGeneration] = (
            asyncio.get_running_loop().create_future()
        )
        self.submitted.append(
            _Submitted(messages, max_new_tokens, sampling, future)
        )
        self.wake.set()
        return future

    def start(self) -> None:
        self.task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        loop = asyncio.get_running_loop()
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None
        # A step already running on the scheduler thread cannot be
        # interrupted; the executor runs this job only after it finishes.
        try:
            await loop.run_in_executor(self.executor, self.scheduler.abort_all)
        except Exception:
            traceback.print_exc(file=sys.stderr)
        for entry in (*self.submitted, *self.handing_over):
            entry.future.cancel()
        for future, _ in self.in_scheduler.values():
            future.cancel()
        self.submitted.clear()
        self.handing_over.clear()
        self.in_scheduler.clear()
        self.executor.shutdown(wait=True)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.submitted and not self.in_scheduler:
                self.wake.clear()
                await self.wake.wait()
                continue
            try:
                await self._iteration(loop)
            except Exception:
                self._recover_from_unexpected_error()
                try:
                    await loop.run_in_executor(
                        self.executor, self.scheduler.abort_all
                    )
                except Exception:
                    traceback.print_exc(file=sys.stderr)

    async def _iteration(self, loop: asyncio.AbstractEventLoop) -> None:
        # Hand over only as many requests as there are free batch slots, so
        # requests still waiting survive a failed step.
        free_slots = self.max_batch_size - len(self.in_scheduler)
        while self.submitted and len(self.handing_over) < free_slots:
            entry = self.submitted.pop(0)
            if not entry.future.done():
                self.handing_over.append(entry)

        outcome = await loop.run_in_executor(
            self.executor, self._worker_iteration, list(self.handing_over)
        )

        for entry, added in zip(self.handing_over, outcome.added, strict=True):
            if isinstance(added, Exception):
                _fail(entry.future, OpenAIRequestError(str(added)))
            else:
                self.in_scheduler[added] = (entry.future, _RequestResult())
        self.handing_over.clear()

        if outcome.step_error is not None:
            # step() already called abort_all(), which dropped every request
            # in the scheduler without events.
            traceback.print_exception(outcome.step_error, file=sys.stderr)
            error = _generation_failed(outcome.step_error)
            for future, _ in self.in_scheduler.values():
                _fail(future, error)
            self.in_scheduler.clear()
            return

        for index, event in outcome.events:
            future, result = self.in_scheduler[index]
            result.add(event)
            if event.finish_reason is None:
                continue
            del self.in_scheduler[index]
            try:
                completed = result.completed()
            except RuntimeError as error:
                _fail(future, _generation_failed(error))
            else:
                _resolve(future, completed)

    def _worker_iteration(self, entries: list[_Submitted]) -> _IterationOutcome:
        """Add new requests, then advance the batch; scheduler thread only."""

        outcome = _IterationOutcome()
        for entry in entries:
            try:
                outcome.added.append(
                    self.scheduler.add_request(
                        entry.messages,
                        max_new_tokens=entry.max_new_tokens,
                        sampling=entry.sampling,
                        enable_thinking=False,
                    )
                )
            except _REQUEST_ERRORS as error:
                outcome.added.append(error)
        if self.scheduler.has_unfinished:
            try:
                outcome.events = self.scheduler.step()
            except Exception as error:
                outcome.step_error = error
        return outcome

    def _recover_from_unexpected_error(self) -> None:
        traceback.print_exc(file=sys.stderr)
        error = OpenAIRequestError(
            "the generation loop failed unexpectedly",
            status_code=500,
            error_type="server_error",
            code="generation_failed",
        )
        for entry in self.handing_over:
            _fail(entry.future, error)
        for future, _ in self.in_scheduler.values():
            _fail(future, error)
        self.handing_over.clear()
        self.in_scheduler.clear()


def _fail(future: asyncio.Future[_CompletedGeneration], error: Exception) -> None:
    if not future.done():
        future.set_exception(error)


def _resolve(
    future: asyncio.Future[_CompletedGeneration], result: _CompletedGeneration
) -> None:
    if not future.done():
        future.set_result(result)


def create_app(
    engine: ServingEngine,
    *,
    served_model: str,
    max_queue_depth: int | None = None,
) -> FastAPI:
    """Create one API application around an already-loaded engine."""

    if not served_model:
        raise ValueError("served_model must not be empty")
    driver = _SchedulerDriver(engine)
    queue_depth = (
        4 * driver.max_batch_size if max_queue_depth is None else max_queue_depth
    )
    if queue_depth < driver.max_batch_size:
        driver.executor.shutdown(wait=False)
        raise ValueError(
            f"max_queue_depth {queue_depth} is below max_batch_size "
            f"{driver.max_batch_size}"
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        driver.start()
        try:
            yield
        finally:
            await driver.stop()

    app = FastAPI(title="mini-llm", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.served_model = served_model
    app.state.max_queue_depth = queue_depth
    app.state.driver = driver

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
            headers=error.headers or None,
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        prepared = prepare_chat_completion_request(
            request, served_model=served_model
        )
        if driver.in_flight >= queue_depth:
            raise OpenAIRequestError(
                f"server is at its queue limit of {queue_depth} requests",
                status_code=429,
                error_type="rate_limit_error",
                code="queue_full",
                headers={"Retry-After": "1"},
            )
        completed = await driver.submit(
            prepared.messages, prepared.max_new_tokens, prepared.sampling
        )
        return create_chat_completion_response(
            model=served_model,
            text=completed.text,
            finish_reason=completed.finish_reason,
            prompt_tokens=completed.prompt_tokens,
            completion_tokens=completed.completion_tokens,
        )

    return app


def _port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be within [1, 65535]")
    return port


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mini_llm.serving.server",
        description="Serve a supported local model through Chat Completions.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port_number, default=8000)
    parser.add_argument("--served-model-name")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--max-batch-size",
        type=_positive_int,
        default=1,
        help="requests decoded together in one continuous batch",
    )
    parser.add_argument(
        "--max-queue-depth",
        type=_positive_int,
        default=None,
        help=(
            "accepted, unfinished requests (running plus waiting) before new "
            "ones get HTTP 429 (default: 4 x --max-batch-size)"
        ),
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
        max_batch_size=args.max_batch_size,
    )
    try:
        # Building the app creates the scheduler, which allocates the KV pool.
        app = create_app(
            engine,
            served_model=served_model,
            max_queue_depth=args.max_queue_depth,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise EngineError(
            f"KV cache of {args.max_seq_len * args.max_batch_size} tokens does "
            "not fit in GPU memory; lower --max-seq-len or --max-batch-size"
        ) from exc
    print(
        f"Loaded on {engine.device} with {engine.dtype} in "
        f"{engine.load_seconds:.2f} s; max batch size "
        f"{app.state.driver.max_batch_size}, queue depth "
        f"{app.state.max_queue_depth}.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_queue_depth is not None and args.max_queue_depth < args.max_batch_size:
        parser.error("--max-queue-depth must be at least --max-batch-size")
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
