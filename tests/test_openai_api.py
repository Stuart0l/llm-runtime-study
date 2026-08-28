from __future__ import annotations

import unittest

from pydantic import ValidationError

from mini_llm.openai_api import (
    ChatCompletionRequest,
    OpenAIRequestError,
    create_chat_completion_response,
    prepare_chat_completion_request,
    to_openai_finish_reason,
    validation_error_response,
)
from mini_llm.tokenizer import ChatMessage


class ChatCompletionRequestTests(unittest.TestCase):
    def test_applies_sampling_and_output_defaults(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        self.assertFalse(request.stream)
        self.assertEqual(request.temperature, 1.0)
        self.assertEqual(request.top_p, 1.0)
        self.assertEqual(request.output_token_limit, 128)
        self.assertEqual(
            request.to_runtime_messages(), [ChatMessage("user", "Hello")]
        )

    def test_concatenates_text_content_parts_and_preserves_history(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "qwen3-0.6b",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is "},
                            {"type": "text", "text": "a KV cache?"},
                        ],
                    },
                    {"role": "assistant", "content": "Stored keys and values."},
                    {"role": "user", "content": "Why keep them?"},
                ],
            }
        )

        self.assertEqual(
            request.to_runtime_messages(),
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("user", "What is a KV cache?"),
                ChatMessage("assistant", "Stored keys and values."),
                ChatMessage("user", "Why keep them?"),
            ],
        )

    def test_resolves_each_supported_output_limit_name(self) -> None:
        common = {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        legacy = ChatCompletionRequest.model_validate(
            {**common, "max_tokens": 17}
        )
        current = ChatCompletionRequest.model_validate(
            {**common, "max_completion_tokens": 23}
        )

        self.assertEqual(legacy.output_token_limit, 17)
        self.assertEqual(current.output_token_limit, 23)

    def test_converts_supported_request_to_runtime_arguments(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "local-qwen",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "top_p": 0.8,
                "seed": 42,
                "max_completion_tokens": 20,
            }
        )

        prepared = prepare_chat_completion_request(
            request, served_model="local-qwen"
        )

        self.assertEqual(prepared.messages, [ChatMessage("user", "Hello")])
        self.assertEqual(prepared.max_new_tokens, 20)
        self.assertEqual(prepared.sampling.temperature, 0.7)
        self.assertEqual(prepared.sampling.top_p, 0.8)
        self.assertEqual(prepared.sampling.seed, 42)

    def test_rejects_streaming_and_stream_options(self) -> None:
        common = {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with self.assertRaisesRegex(ValidationError, "streaming is not supported"):
            ChatCompletionRequest.model_validate({**common, "stream": True})

        with self.assertRaises(ValidationError):
            ChatCompletionRequest.model_validate(
                {**common, "stream_options": {"include_usage": True}}
            )

    def test_rejects_conflicting_token_limits(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cannot both be supplied"):
            ChatCompletionRequest.model_validate(
                {
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                    "max_completion_tokens": 20,
                }
            )

    def test_rejects_unsupported_roles_content_and_fields(self) -> None:
        invalid_requests = [
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "developer", "content": "Hello"}],
            },
            {
                "model": "qwen3-0.6b",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {}}],
                    }
                ],
            },
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": []}],
            },
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [],
            },
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello"}],
                "frequency_penalty": 0.5,
            },
        ]

        for payload in invalid_requests:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                ChatCompletionRequest.model_validate(payload)

    def test_rejects_invalid_role_order_and_final_assistant(self) -> None:
        for messages in (
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        ):
            with self.subTest(messages=messages), self.assertRaises(ValidationError):
                ChatCompletionRequest.model_validate(
                    {"model": "qwen3-0.6b", "messages": messages}
                )

    def test_rejects_invalid_sampling_choice_and_limits(self) -> None:
        common = {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        invalid_fields = (
            {"temperature": -1},
            {"temperature": 3},
            {"top_p": 0},
            {"top_p": 1.1},
            {"seed": -1},
            {"max_tokens": 0},
            {"max_completion_tokens": 0},
            {"n": 2},
        )

        for fields in invalid_fields:
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                ChatCompletionRequest.model_validate({**common, **fields})

    def test_unknown_model_has_openai_error_body(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "unknown",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

        with self.assertRaises(OpenAIRequestError) as caught:
            prepare_chat_completion_request(request, served_model="local-qwen")

        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(
            caught.exception.response.model_dump(),
            {
                "error": {
                    "message": "The model 'unknown' does not exist",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )


class ChatCompletionResponseTests(unittest.TestCase):
    def test_builds_exact_non_streaming_response_shape(self) -> None:
        response = create_chat_completion_response(
            model="qwen3-0.6b",
            text="Hello!",
            finish_reason="eos",
            prompt_tokens=12,
            completion_tokens=3,
            completion_id="chatcmpl-test",
            created=123,
        )

        self.assertEqual(
            response.model_dump(),
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 123,
                "model": "qwen3-0.6b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                },
            },
        )

    def test_maps_all_runtime_finish_reasons(self) -> None:
        self.assertEqual(to_openai_finish_reason("eos"), "stop")
        self.assertEqual(to_openai_finish_reason("max_new_tokens"), "length")
        self.assertEqual(to_openai_finish_reason("context_length"), "length")

    def test_converts_schema_error_to_openai_envelope(self) -> None:
        try:
            ChatCompletionRequest.model_validate(
                {
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "n": 2,
                }
            )
        except ValidationError as error:
            response = validation_error_response(error)
        else:  # pragma: no cover - protects the test itself
            self.fail("expected request validation to fail")

        self.assertEqual(
            response.model_dump(),
            {
                "error": {
                    "message": "only n=1 is supported",
                    "type": "invalid_request_error",
                    "param": "n",
                    "code": None,
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
