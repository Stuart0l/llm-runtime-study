from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import Iterator, Sequence

import httpx

from mini_llm.generation import FinishReason, GenerationError, GenerationEvent
from mini_llm.sampling import SamplingConfig
from mini_llm.server import create_app
from mini_llm.tokenizer import ChatMessage


class _FakeEngine:
    def __init__(
        self,
        *,
        finish_reason: FinishReason = "eos",
        delay: float = 0.0,
        failure: Exception | None = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.delay = delay
        self.failure = failure
        self.calls: list[tuple[list[ChatMessage], int, SamplingConfig, bool]] = []
        self.started: list[str] = []
        self.active = 0
        self.max_active = 0
        self._state_lock = threading.Lock()

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig,
        enable_thinking: bool,
    ) -> Iterator[GenerationEvent]:
        self.calls.append((list(messages), max_new_tokens, sampling, enable_thinking))
        if self.failure is not None:
            raise self.failure

        prompt = messages[-1].content

        def iterate() -> Iterator[GenerationEvent]:
            with self._state_lock:
                self.started.append(prompt)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(self.delay)
                yield GenerationEvent(
                    token_id=10,
                    token_index=0,
                    text_delta=f"reply:{prompt}",
                    text=f"reply:{prompt}",
                    finish_reason=self.finish_reason,
                    prompt_token_count=7,
                )
            finally:
                with self._state_lock:
                    self.active -= 1

        return iterate()


class ChatCompletionsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = _FakeEngine()
        self.app = create_app(self.engine, served_model="local-qwen")
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_returns_openai_chat_completion_and_usage(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
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

        messages, limit, sampling, thinking = self.engine.calls[0]
        self.assertEqual(
            messages,
            [ChatMessage("system", "Be brief."), ChatMessage("user", "hello")],
        )
        self.assertEqual(limit, 9)
        self.assertEqual(sampling, SamplingConfig(temperature=0, top_p=0.8, seed=3))
        self.assertFalse(thinking)

    async def test_application_owns_the_supplied_engine_and_one_lock(self) -> None:
        self.assertIs(self.app.state.engine, self.engine)
        self.assertEqual(self.app.state.served_model, "local-qwen")
        self.assertIsInstance(self.app.state.generation_lock, asyncio.Lock)

    async def test_maps_token_limit_to_length(self) -> None:
        engine = _FakeEngine(finish_reason="max_new_tokens")
        app = create_app(engine, served_model="local-qwen")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    async def test_rejects_invalid_request_with_openai_error_shape(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

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
        self.assertEqual(self.engine.calls, [])

    async def test_rejects_malformed_json_with_openai_error_shape(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            content=b'{"model":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertIn("JSON", response.json()["error"]["message"])
        self.assertEqual(self.engine.calls, [])

    async def test_rejects_unknown_model_with_openai_error_shape(self) -> None:
        response = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": "not-loaded",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["param"], "model")
        self.assertEqual(response.json()["error"]["code"], "model_not_found")
        self.assertEqual(self.engine.calls, [])

    async def test_maps_runtime_request_failure_to_openai_error(self) -> None:
        engine = _FakeEngine(failure=GenerationError("prompt exceeds context"))
        app = create_app(engine, served_model="local-qwen")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-qwen",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["message"], "prompt exceeds context")

    async def test_serializes_concurrent_requests_using_shared_cache(self) -> None:
        engine = _FakeEngine(delay=0.05)
        app = create_app(engine, served_model="local-qwen")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-qwen",
                        "messages": [{"role": "user", "content": "first"}],
                    },
                )
            )
            await asyncio.sleep(0)
            second = asyncio.create_task(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-qwen",
                        "messages": [{"role": "user", "content": "second"}],
                    },
                )
            )
            responses = await asyncio.gather(first, second)

        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(engine.started, ["first", "second"])
        self.assertEqual(engine.max_active, 1)


class ApplicationConstructionTests(unittest.TestCase):
    def test_rejects_empty_served_model_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            create_app(_FakeEngine(), served_model="")


if __name__ == "__main__":
    unittest.main()
