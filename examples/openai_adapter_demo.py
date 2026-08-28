"""Print representative OpenAI-compatible request, response, and error JSON."""

from __future__ import annotations

import json

from pydantic import ValidationError

from mini_llm.openai_api import (
    ChatCompletionRequest,
    create_chat_completion_response,
    prepare_chat_completion_request,
    validation_error_response,
)


def _print_json(title: str, value: object) -> None:
    print(f"{title}:\n{json.dumps(value, indent=2)}\n")


def main() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen3-0.6b",
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is "},
                        {"type": "text", "text": "a KV cache?"},
                    ],
                },
            ],
            "temperature": 0,
            "max_completion_tokens": 32,
        }
    )
    prepared = prepare_chat_completion_request(
        request, served_model="qwen3-0.6b"
    )
    _print_json("Validated request", request.model_dump(exclude_none=True))
    print(f"Runtime messages: {prepared.messages}")
    print(f"Runtime output limit: {prepared.max_new_tokens}\n")

    response = create_chat_completion_response(
        model="qwen3-0.6b",
        text="A KV cache stores attention keys and values.",
        finish_reason="eos",
        prompt_tokens=24,
        completion_tokens=10,
        completion_id="chatcmpl-demo",
        created=0,
    )
    _print_json("Success response", response.model_dump())

    try:
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen3-0.6b",
                "messages": [{"role": "user", "content": "Hello"}],
                "n": 2,
            }
        )
    except ValidationError as error:
        _print_json("Error response", validation_error_response(error).model_dump())


if __name__ == "__main__":
    main()
