"""Architecture-specific text chat templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from mini_llm.interfaces import ChatMessage


class TokenizerError(ValueError):
    """Raised when tokenizer artifacts or chat messages are invalid."""


def validate_chat_messages(
    messages: Sequence[ChatMessage], *, add_generation_prompt: bool
) -> None:
    """Validate the text-only conversation structure supported by the runtime."""

    if not messages:
        raise TokenizerError("chat requires at least one message")

    expected_role = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, ChatMessage):
            raise TokenizerError(
                f"message {index} must be ChatMessage, got {type(message).__name__}"
            )
        if not isinstance(message.content, str):
            raise TokenizerError(f"message {index} content must be str")
        if message.role == "system":
            if index != 0:
                raise TokenizerError(
                    "a system message is only allowed as the first turn"
                )
            continue
        if message.role not in ("user", "assistant"):
            raise TokenizerError(
                f"unsupported chat role at message {index}: {message.role!r}"
            )
        if message.role != expected_role:
            raise TokenizerError(
                f"expected {expected_role!r} at message {index}, got {message.role!r}"
            )
        expected_role = "assistant" if message.role == "user" else "user"

    if add_generation_prompt and messages[-1].role != "user":
        raise TokenizerError(
            "the final message must have role 'user' when adding a generation prompt"
        )


class ChatTemplate(ABC):
    """Formatting contract implemented independently by each architecture."""

    @classmethod
    @abstractmethod
    def format(
        cls,
        messages: Sequence[ChatMessage],
        *,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Serialize a validated conversation into model input text."""


@dataclass(frozen=True, slots=True)
class Qwen3ChatTokens:
    """Textual control tokens belonging to Qwen3's chat protocol."""

    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    end_of_text: str = "<|endoftext|>"
    think_start: str = "<think>"
    think_end: str = "</think>"


class Qwen3ChatTemplate(ChatTemplate):
    """Qwen3 role boundaries and optional thinking prefix."""

    tokens = Qwen3ChatTokens()

    @classmethod
    def format(
        cls,
        messages: Sequence[ChatMessage],
        *,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        validate_chat_messages(
            messages, add_generation_prompt=add_generation_prompt
        )
        tokens = cls.tokens
        prompt = "".join(
            f"{tokens.im_start}{message.role}\n{message.content}{tokens.im_end}\n"
            for message in messages
        )
        if add_generation_prompt:
            prompt += f"{tokens.im_start}assistant\n"
            if not enable_thinking:
                prompt += f"{tokens.think_start}\n\n{tokens.think_end}\n\n"
        return prompt


@dataclass(frozen=True, slots=True)
class GraniteChatTokens:
    """Control tokens used by Granite's role-based chat protocol."""

    end_of_text: str = "<|end_of_text|>"
    start_of_role: str = "<|start_of_role|>"
    end_of_role: str = "<|end_of_role|>"
    tool_call: str = "<|tool_call|>"


class GraniteChatTemplate(ChatTemplate):
    """Granite role boundaries and official default system message."""

    tokens = GraniteChatTokens()

    @classmethod
    def format(
        cls,
        messages: Sequence[ChatMessage],
        *,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
        today: date | None = None,
    ) -> str:
        if enable_thinking:
            raise TokenizerError("thinking mode is supported by Qwen3, not Granite")
        validate_chat_messages(
            messages, add_generation_prompt=add_generation_prompt
        )

        if messages[0].role == "system":
            system_message = messages[0].content
            conversation = messages[1:]
        else:
            current_date = today or date.today()
            system_message = (
                "Knowledge Cutoff Date: April 2024.\n"
                f"Today's Date: {current_date.strftime('%B %d, %Y')}.\n"
                "You are Granite, developed by IBM. You are a helpful AI assistant."
            )
            conversation = messages

        tokens = cls.tokens

        def format_message(role: str, content: str) -> str:
            return (
                f"{tokens.start_of_role}{role}{tokens.end_of_role}"
                f"{content}{tokens.end_of_text}\n"
            )

        prompt = format_message("system", system_message)
        prompt += "".join(
            format_message(message.role, message.content)
            for message in conversation
        )
        if add_generation_prompt:
            prompt += f"{tokens.start_of_role}assistant{tokens.end_of_role}"
        return prompt
