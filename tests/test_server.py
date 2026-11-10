from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
import threading
import unittest
from typing import AsyncIterator, Callable, Sequence
from unittest.mock import ANY, MagicMock, patch

from asgi_lifespan import LifespanManager
import httpx
import torch

from tests.reference_support import has_local_checkpoint

from mini_llm.engine import Engine, EngineError
from mini_llm.generation import FinishReason, GenerationError, GenerationEvent
from mini_llm.sampling import SamplingConfig, SamplingError
from mini_llm.serving import server
from mini_llm.serving.server import build_parser, create_app, main
from mini_llm.tokenizer import ChatMessage


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"
URL = "/v1/chat/completions"


class _FakeScheduler:
    """A model-free ServingScheduler with hooks for timing and failures."""

    def __init__(
        self,
        *,
        max_batch_size: int = 1,
        steps_to_finish: int = 1,
        finish_reason: FinishReason = "eos",
        add_request_error: Exception | None = None,
        step_error: Exception | None = None,
        hold_step: threading.Event | None = None,
        scripted_events: list[list[tuple[int, GenerationEvent]]] | None = None,
        abort_error: Exception | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.steps_to_finish = steps_to_finish
        self.finish_reason = finish_reason
        self.add_request_error = add_request_error
        self.step_error = step_error
        self.hold_step = hold_step
        self.scripted_events = scripted_events
        self.abort_error = abort_error
        self.log: list[tuple[str, threading.Thread]] = []
        self.calls: list[tuple[list[ChatMessage], int, SamplingConfig, bool]] = []
        self.max_running = 0
        self.aborts = 0
        self._next_index = 0
        # index -> [prompt, steps taken]
        self._running: dict[int, list] = {}

    def _record(self, name: str) -> None:
        self.log.append((name, threading.current_thread()))

    @property
    def has_unfinished(self) -> bool:
        return bool(self._running)

    def add_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
        enable_thinking: bool,
    ) -> int:
        self._record("add_request")
        self.calls.append((list(messages), max_new_tokens, sampling, enable_thinking))
        if self.add_request_error is not None:
            error, self.add_request_error = self.add_request_error, None
            raise error
        index = self._next_index
        self._next_index += 1
        self._running[index] = [messages[-1].content, 0]
        return index

    def step(self) -> list[tuple[int, GenerationEvent]]:
        self._record("step")
        if self.hold_step is not None:
            self.hold_step.wait(timeout=30)
        if self.step_error is not None:
            error, self.step_error = self.step_error, None
            self._running.clear()
            raise error
        self.max_running = max(self.max_running, len(self._running))
        if self.scripted_events:
            events = self.scripted_events.pop(0)
            for index, event in events:
                if event.finish_reason is not None:
                    self._running.pop(index, None)
            return events
        events = []
        for index, state in list(self._running.items()):
            prompt, steps = state
            state[1] = steps + 1
            finished = state[1] >= self.steps_to_finish
            events.append(
                (
                    index,
                    GenerationEvent(
                        token_id=10,
                        token_index=steps,
                        text_delta="x",
                        text=f"reply:{prompt}",
                        finish_reason=self.finish_reason if finished else None,
                        prompt_token_count=7 if steps == 0 else None,
                    ),
                )
            )
            if finished:
                del self._running[index]
        return events

    def abort_all(self) -> None:
        self._record("abort_all")
        self.aborts += 1
        self._running.clear()
        if self.abort_error is not None:
            error, self.abort_error = self.abort_error, None
            raise error


class _FakeEngine:
    def __init__(self, scheduler: _FakeScheduler | None = None) -> None:
        self.scheduler = scheduler or _FakeScheduler()
        self.create_calls = 0

    def create_scheduler(self) -> _FakeScheduler:
        self.create_calls += 1
        return self.scheduler


@asynccontextmanager
async def _serving(app) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app),
            base_url="http://testserver",
        ) as client:
            yield client


async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.005)


def _body(content: str = "hello", **extra) -> dict:
    return {
        "model": "local-qwen",
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


class ChatCompletionsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _FakeEngine()
        self.app = create_app(self.engine, served_model="local-qwen")
        self.stack = AsyncExitStack()
        self.client = await self.stack.enter_async_context(_serving(self.app))

    async def asyncTearDown(self) -> None:
        await self.stack.aclose()

    async def test_returns_openai_chat_completion_and_usage(self) -> None:
        response = await self.client.post(
            URL,
            json={
                "model": "local-qwen",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hello"},
                ],
                "temperature": 0,
                "top_p": 0.8,
                "seed": 3,
                "max_completion_tokens": 9,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertIsInstance(body["created"], int)
        self.assertEqual(body["model"], "local-qwen")
        self.assertEqual(
            body["choices"],
            [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "reply:hello",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        )
        self.assertEqual(
            body["usage"],
            {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        )

        messages, limit, sampling, thinking = self.engine.scheduler.calls[0]
        self.assertEqual(
            messages,
            [ChatMessage("system", "Be brief."), ChatMessage("user", "hello")],
        )
        self.assertEqual(limit, 9)
        self.assertEqual(sampling, SamplingConfig(temperature=0, top_p=0.8, seed=3))
        self.assertFalse(thinking)

    async def test_rejects_invalid_request_with_openai_error_shape(self) -> None:
        response = await self.client.post(URL, json=_body(stream=True))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "message": "streaming is not supported",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": None,
                }
            },
        )
        self.assertEqual(self.engine.scheduler.calls, [])

    async def test_rejects_malformed_json_with_openai_error_shape(self) -> None:
        response = await self.client.post(
            URL,
            content=b'{"model":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertIn("JSON", response.json()["error"]["message"])
        self.assertEqual(self.engine.scheduler.calls, [])

    async def test_rejects_unknown_model_with_openai_error_shape(self) -> None:
        response = await self.client.post(
            URL,
            json={
                "model": "not-loaded",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["param"], "model")
        self.assertEqual(response.json()["error"]["code"], "model_not_found")
        self.assertEqual(self.engine.scheduler.calls, [])

    async def test_zero_token_context_length_finish(self) -> None:
        self.engine.scheduler.scripted_events = [
            [
                (
                    0,
                    GenerationEvent(
                        token_id=None,
                        token_index=None,
                        text_delta="",
                        text="",
                        finish_reason="context_length",
                        prompt_token_count=7,
                    ),
                )
            ]
        ]

        response = await self.client.post(URL, json=_body())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 0)


class BatchedServingTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_prompt_check_failure_to_400_and_keeps_serving(self) -> None:
        scheduler = _FakeScheduler(
            add_request_error=GenerationError("prompt exceeds context")
        )
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        async with _serving(app) as client:
            failed = await client.post(URL, json=_body())
            succeeded = await client.post(URL, json=_body())

        self.assertEqual(failed.status_code, 400)
        self.assertEqual(failed.json()["error"]["message"], "prompt exceeds context")
        self.assertEqual(failed.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(succeeded.status_code, 200)

    async def test_concurrent_requests_share_one_batch_on_one_thread(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(
            max_batch_size=4, steps_to_finish=2, hold_step=event
        )
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        async with _serving(app) as client:
            first = asyncio.create_task(client.post(URL, json=_body("a")))
            await _wait_for(lambda: len(scheduler.calls) == 1)
            second = asyncio.create_task(client.post(URL, json=_body("b")))
            await _wait_for(lambda: app.state.driver.in_flight == 2)
            event.set()
            responses = await asyncio.gather(first, second)

        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(
            [r.json()["choices"][0]["message"]["content"] for r in responses],
            ["reply:a", "reply:b"],
        )
        self.assertEqual(scheduler.max_running, 2)
        self.assertEqual(len({thread for _, thread in scheduler.log}), 1)
        self.assertIsNot(scheduler.log[0][1], threading.current_thread())

    async def test_queue_limit_counts_requests_being_handed_over(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(max_batch_size=2, hold_step=event)
        app = create_app(
            _FakeEngine(scheduler), served_model="local-qwen", max_queue_depth=2
        )
        async with _serving(app) as client:
            first = asyncio.create_task(client.post(URL, json=_body("a")))
            await _wait_for(lambda: len(scheduler.calls) == 1)
            second = asyncio.create_task(client.post(URL, json=_body("b")))
            # A is inside the held scheduler job (handing_over) and B waits in
            # the driver queue; both must count toward the limit.
            await _wait_for(lambda: app.state.driver.in_flight == 2)
            self.assertEqual(len(app.state.driver.handing_over), 1)
            rejected = await client.post(URL, json=_body("c"))
            event.set()
            accepted = await asyncio.gather(first, second)

        self.assertEqual(rejected.status_code, 429)
        self.assertEqual(rejected.headers["retry-after"], "1")
        self.assertEqual(rejected.json()["error"]["type"], "rate_limit_error")
        self.assertEqual(rejected.json()["error"]["code"], "queue_full")
        self.assertEqual([r.status_code for r in accepted], [200, 200])
        self.assertEqual(len(scheduler.calls), 2)

    async def test_failed_step_returns_500_and_server_recovers(self) -> None:
        for error in (RuntimeError("boom"), SamplingError("non-finite logits")):
            with self.subTest(error=type(error).__name__):
                scheduler = _FakeScheduler(step_error=error)
                app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
                with redirect_stderr(StringIO()):
                    async with _serving(app) as client:
                        failed = await client.post(URL, json=_body())
                        succeeded = await client.post(URL, json=_body())

                self.assertEqual(failed.status_code, 500)
                self.assertEqual(
                    failed.json()["error"],
                    {
                        "message": str(error),
                        "type": "server_error",
                        "param": None,
                        "code": "generation_failed",
                    },
                )
                self.assertEqual(succeeded.status_code, 200)

    async def test_bookkeeping_error_clears_state_and_survives_abort_failure(
        self,
    ) -> None:
        original_add = server._RequestResult.add
        for abort_error in (None, RuntimeError("abort failed")):
            with self.subTest(abort_error=abort_error):
                scheduler = _FakeScheduler(abort_error=abort_error)
                app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
                failures = iter([RuntimeError("bookkeeping bug")])

                def add(result, event):
                    error = next(failures, None)
                    if error is not None:
                        raise error
                    original_add(result, event)

                with (
                    patch.object(server._RequestResult, "add", add),
                    redirect_stderr(StringIO()),
                ):
                    async with _serving(app) as client:
                        failed = await client.post(URL, json=_body())
                        self.assertFalse(app.state.driver.task.done())
                        # max_batch_size is 1: this only succeeds if the
                        # failed request's slot was freed.
                        succeeded = await client.post(URL, json=_body())

                self.assertEqual(failed.status_code, 500)
                self.assertEqual(failed.json()["error"]["code"], "generation_failed")
                self.assertEqual(succeeded.status_code, 200)
                self.assertGreaterEqual(scheduler.aborts, 1)

    async def test_cancelled_request_does_not_stop_the_driver(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(hold_step=event)
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        async with _serving(app) as client:
            abandoned = asyncio.create_task(client.post(URL, json=_body("a")))
            await _wait_for(lambda: len(scheduler.calls) == 1)
            abandoned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await abandoned
            event.set()
            response = await client.post(URL, json=_body("b"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "reply:b")

    async def test_waiting_requests_wait_for_a_free_slot_in_order(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(max_batch_size=1, hold_step=event)
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        async with _serving(app) as client:
            tasks = [asyncio.create_task(client.post(URL, json=_body("a")))]
            await _wait_for(lambda: len(scheduler.calls) == 1)
            tasks.append(asyncio.create_task(client.post(URL, json=_body("b"))))
            tasks.append(asyncio.create_task(client.post(URL, json=_body("c"))))
            await _wait_for(lambda: app.state.driver.in_flight == 3)
            # Only one slot exists, so B and C stay in the driver queue.
            self.assertEqual(len(scheduler.calls), 1)
            event.set()
            responses = await asyncio.gather(*tasks)

        self.assertEqual([r.status_code for r in responses], [200, 200, 200])
        self.assertEqual(
            [call[0][-1].content for call in scheduler.calls], ["a", "b", "c"]
        )

    async def test_waiting_request_survives_a_failed_step(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(
            max_batch_size=1, hold_step=event, step_error=RuntimeError("boom")
        )
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        with redirect_stderr(StringIO()):
            async with _serving(app) as client:
                first = asyncio.create_task(client.post(URL, json=_body("a")))
                await _wait_for(lambda: len(scheduler.calls) == 1)
                second = asyncio.create_task(client.post(URL, json=_body("b")))
                await _wait_for(lambda: app.state.driver.in_flight == 2)
                event.set()
                failed, succeeded = await asyncio.gather(first, second)

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(succeeded.status_code, 200)
        names = [name for name, _ in scheduler.log]
        self.assertEqual(names[:3], ["add_request", "step", "add_request"])

    async def test_shutdown_waits_for_the_running_step_before_aborting(self) -> None:
        event = threading.Event()
        scheduler = _FakeScheduler(hold_step=event)
        app = create_app(_FakeEngine(scheduler), served_model="local-qwen")
        driver = app.state.driver
        driver.start()
        future = driver.submit(
            [ChatMessage("user", "a")], 8, SamplingConfig(temperature=0)
        )
        await _wait_for(lambda: [n for n, _ in scheduler.log] == ["add_request", "step"])

        stop = asyncio.create_task(driver.stop())
        threading.Timer(0.2, event.set).start()
        await stop

        (step_name, step_thread), (abort_name, abort_thread) = scheduler.log[-2:]
        self.assertEqual((step_name, abort_name), ("step", "abort_all"))
        self.assertIs(step_thread, abort_thread)
        self.assertTrue(future.cancelled())


class ApplicationConstructionTests(unittest.TestCase):
    def test_rejects_empty_served_model_name(self) -> None:
        engine = _FakeEngine()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            create_app(engine, served_model="")
        self.assertEqual(engine.create_calls, 0)

    def test_rejects_queue_depth_below_batch_size(self) -> None:
        engine = _FakeEngine(_FakeScheduler(max_batch_size=4))
        with self.assertRaisesRegex(ValueError, "below max_batch_size"):
            create_app(engine, served_model="local-qwen", max_queue_depth=2)


def _cli_engine(**attributes) -> MagicMock:
    engine = MagicMock(**attributes)
    engine.create_scheduler.return_value = _FakeScheduler()
    return engine


class ServerCommandTests(unittest.TestCase):
    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_loads_engine_once_before_starting_one_worker(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        engine = _cli_engine(
            device=torch.device("cpu"), dtype=torch.float32, load_seconds=1.25
        )
        from_model_dir.return_value = engine
        output = StringIO()

        status = main(
            [
                "--model",
                "models/qwen3-0.6b",
                "--host",
                "127.0.0.1",
                "--port",
                "8123",
                "--device",
                "cuda",
                "--dtype",
                "float32",
                "--max-seq-len",
                "2048",
            ],
            output=output,
        )

        self.assertEqual(status, 0)
        from_model_dir.assert_called_once_with(
            Path("models/qwen3-0.6b"),
            device="cuda",
            dtype="float32",
            max_seq_len=2048,
            max_batch_size=1,
        )
        uvicorn_run.assert_called_once_with(
            ANY,
            host="127.0.0.1",
            port=8123,
            workers=1,
        )
        app = uvicorn_run.call_args.args[0]
        self.assertIs(app.state.engine, engine)
        self.assertEqual(app.state.served_model, "qwen3-0.6b")
        self.assertEqual(app.state.max_queue_depth, 4)
        self.assertIn("Loaded on cpu", output.getvalue())
        self.assertIn("max batch size 1, queue depth 4", output.getvalue())

    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_uses_explicit_served_model_name(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        from_model_dir.return_value = _cli_engine(
            device="mps", dtype="float16", load_seconds=1.0
        )

        status = main(
            [
                "--model",
                "model-dir",
                "--served-model-name",
                "my-local-model",
            ],
            output=StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(
            uvicorn_run.call_args.args[0].state.served_model,
            "my-local-model",
        )

    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_passes_batch_size_and_queue_depth(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        engine = _cli_engine(device="cpu", dtype="float32", load_seconds=1.0)
        engine.create_scheduler.return_value = _FakeScheduler(max_batch_size=3)
        from_model_dir.return_value = engine

        status = main(
            ["--model", "model-dir", "--max-batch-size", "3", "--max-queue-depth", "6"],
            output=StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(from_model_dir.call_args.kwargs["max_batch_size"], 3)
        self.assertEqual(uvicorn_run.call_args.args[0].state.max_queue_depth, 6)

    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_rejects_queue_depth_below_batch_size_before_loading(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            main(
                ["--model", "model-dir", "--max-batch-size", "4", "--max-queue-depth", "2"],
                output=StringIO(),
            )

        self.assertEqual(raised.exception.code, 2)
        from_model_dir.assert_not_called()
        uvicorn_run.assert_not_called()

    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_reports_kv_pool_out_of_memory_without_starting_server(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        engine = _cli_engine(device="cuda", dtype="float16", load_seconds=1.0)
        engine.create_scheduler.side_effect = torch.cuda.OutOfMemoryError("oom")
        from_model_dir.return_value = engine
        error = StringIO()

        status = main(
            ["--model", "model-dir", "--max-seq-len", "4096", "--max-batch-size", "16"],
            output=StringIO(),
            error=error,
        )

        self.assertEqual(status, 2)
        self.assertEqual(
            error.getvalue(),
            "error: KV cache of 65536 tokens does not fit in GPU memory; "
            "lower --max-seq-len or --max-batch-size\n",
        )
        uvicorn_run.assert_not_called()

    @patch("mini_llm.serving.server.uvicorn.run")
    @patch("mini_llm.serving.server.Engine.from_model_dir")
    def test_reports_model_loading_error_without_starting_server(
        self, from_model_dir: MagicMock, uvicorn_run: MagicMock
    ) -> None:
        from_model_dir.side_effect = EngineError("MPS is unavailable")
        error = StringIO()

        status = main(["--model", "model-dir"], output=StringIO(), error=error)

        self.assertEqual(status, 2)
        self.assertEqual(error.getvalue(), "error: MPS is unavailable\n")
        uvicorn_run.assert_not_called()

    def test_rejects_port_outside_tcp_range(self) -> None:
        parser = build_parser()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--model", "model-dir", "--port", "70000"])


class RealCheckpointHTTPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_real_cpu_checkpoint_serves(
        self, model_dir: Path
    ) -> None:
        engine = await asyncio.to_thread(
            Engine.from_model_dir,
            model_dir,
            device="cpu",
            dtype="bfloat16",
            max_seq_len=128,
        )
        served_model = model_dir.name
        app = create_app(engine, served_model=served_model)
        async with _serving(app) as client:
            response = await client.post(
                URL,
                json={
                    "model": served_model,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "temperature": 0,
                    "max_completion_tokens": 1,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], served_model)
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertIsInstance(body["choices"][0]["message"]["content"], str)
        self.assertEqual(body["usage"]["completion_tokens"], 1)
        self.assertEqual(
            body["usage"]["total_tokens"],
            body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"],
        )

    @unittest.skipUnless(
        has_local_checkpoint(QWEN_MODEL_DIR), "local Qwen3 checkpoint is unavailable"
    )
    async def test_real_qwen_cpu_checkpoint_serves_chat_completion(self) -> None:
        await self._assert_real_cpu_checkpoint_serves(QWEN_MODEL_DIR)

    @unittest.skipUnless(
        has_local_checkpoint(GRANITE_MODEL_DIR), "local Granite checkpoint is unavailable"
    )
    async def test_real_granite_cpu_checkpoint_serves_chat_completion(self) -> None:
        await self._assert_real_cpu_checkpoint_serves(GRANITE_MODEL_DIR)


if __name__ == "__main__":
    unittest.main()
