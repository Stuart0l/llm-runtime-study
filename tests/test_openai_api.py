from __future__ import annotations

import unittest

from pydantic import ValidationError

from mini_llm.serving.openai_api import (
    ChatCompletionRequest,
    to_openai_finish_reason,
)
from mini_llm.tokenizer import ChatMessage


class ChatCompletionRequestTests(unittest.TestCase):
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

    def test_rejects_unsupported_payloads(self) -> None:
        common = {
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        invalid_requests = [
            (
                {
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "developer", "content": "Hello"}],
                },
                None,
            ),
            (
                {
                    "model": "qwen3-0.6b",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {}}],
                        }
                    ],
                },
                None,
            ),
            (
                {
                    "model": "qwen3-0.6b",
                    "messages": [{"role": "user", "content": []}],
                },
                None,
            ),
            ({**common, "tools": []}, None),
            ({**common, "frequency_penalty": 0.5}, None),
            ({**common, "stream_options": {"include_usage": True}}, None),
            (
                {**common, "max_tokens": 10, "max_completion_tokens": 20},
                "cannot both be supplied",
            ),
            (
                {
                    "model": "qwen3-0.6b",
                    "messages": [
                        {"role": "user", "content": "one"},
                        {"role": "user", "content": "two"},
                    ],
                },
                None,
            ),
            (
                {
                    "model": "qwen3-0.6b",
                    "messages": [
                        {"role": "user", "content": "one"},
                        {"role": "assistant", "content": "two"},
                    ],
                },
                None,
            ),
            ({**common, "temperature": -1}, None),
            ({**common, "temperature": 3}, None),
            ({**common, "top_p": 0}, None),
            ({**common, "top_p": 1.1}, None),
            ({**common, "seed": -1}, None),
            ({**common, "max_tokens": 0}, None),
            ({**common, "max_completion_tokens": 0}, None),
            ({**common, "n": 2}, None),
        ]

        for payload, message in invalid_requests:
            with self.subTest(payload=payload):
                if message is None:
                    with self.assertRaises(ValidationError):
                        ChatCompletionRequest.model_validate(payload)
                else:
                    with self.assertRaisesRegex(ValidationError, message):
                        ChatCompletionRequest.model_validate(payload)


class ChatCompletionResponseTests(unittest.TestCase):
    def test_maps_all_runtime_finish_reasons(self) -> None:
        self.assertEqual(to_openai_finish_reason("eos"), "stop")
        self.assertEqual(to_openai_finish_reason("max_new_tokens"), "length")
        self.assertEqual(to_openai_finish_reason("context_length"), "length")


if __name__ == "__main__":
    unittest.main()
