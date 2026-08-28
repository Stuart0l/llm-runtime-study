"""Qwen3 text tokenization and minimal chat prompt formatting.

The ``tokenizers`` package owns the trained BPE vocabulary and segmentation
algorithm.  This module owns the runtime-facing contract: locating the local
artifacts, validating Qwen control tokens, and building the prompt protocol
that the instruction-tuned model expects.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Sequence

from tokenizers import Tokenizer

from mini_llm.config import Qwen3Config


class TokenizerError(ValueError):
    """Raised when tokenizer artifacts or chat messages are invalid."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One textual message in the minimal Qwen3 chat protocol."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Qwen3ChatTokens:
    """Textual control tokens belonging to Qwen3's chat protocol."""

    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    end_of_text: str = "<|endoftext|>"
    think_start: str = "<think>"
    think_end: str = "</think>"


QWEN3_CHAT_TOKENS = Qwen3ChatTokens()


@dataclass(frozen=True, slots=True)
class Qwen3SpecialTokenIds:
    """IDs of the control tokens needed by the first runtime version."""

    end_of_text: int
    im_start: int
    im_end: int
    think_start: int
    think_end: int


class Qwen3Tokenizer:
    """Thin, validated wrapper around a local Qwen tokenizer."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        special_tokens: Qwen3SpecialTokenIds,
        model_vocab_size: int,
    ) -> None:
        self._tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.model_vocab_size = model_vocab_size

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3Tokenizer":
        model_path = Path(model_dir)
        tokenizer_path = model_path / "tokenizer.json"
        tokenizer_config_path = model_path / "tokenizer_config.json"
        if not tokenizer_path.is_file():
            raise TokenizerError(f"tokenizer file does not exist: {tokenizer_path}")
        if not tokenizer_config_path.is_file():
            raise TokenizerError(
                f"tokenizer configuration does not exist: {tokenizer_config_path}"
            )

        model_config = Qwen3Config.from_model_dir(model_path)
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise TokenizerError(
                f"invalid JSON in {tokenizer_config_path}: {exc}"
            ) from exc
        if not isinstance(tokenizer_config, dict):
            raise TokenizerError(
                f"tokenizer configuration root must be an object: {tokenizer_config_path}"
            )
        chat_tokens = QWEN3_CHAT_TOKENS
        if tokenizer_config.get("eos_token") != chat_tokens.im_end:
            raise TokenizerError(f"Qwen3 eos_token must be {chat_tokens.im_end!r}")
        if tokenizer_config.get("pad_token") != chat_tokens.end_of_text:
            raise TokenizerError(
                f"Qwen3 pad_token must be {chat_tokens.end_of_text!r}"
            )

        try:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise TokenizerError(f"could not load tokenizer: {tokenizer_path}") from exc

        special_tokens = Qwen3SpecialTokenIds(
            end_of_text=cls._required_token_id(tokenizer, chat_tokens.end_of_text),
            im_start=cls._required_token_id(tokenizer, chat_tokens.im_start),
            im_end=cls._required_token_id(tokenizer, chat_tokens.im_end),
            think_start=cls._required_token_id(tokenizer, chat_tokens.think_start),
            think_end=cls._required_token_id(tokenizer, chat_tokens.think_end),
        )
        if special_tokens.end_of_text != model_config.bos_token_id:
            raise TokenizerError(
                "model bos_token_id does not match the tokenizer's end-of-text token: "
                f"{model_config.bos_token_id} != {special_tokens.end_of_text}"
            )
        if special_tokens.im_end not in model_config.eos_token_ids:
            raise TokenizerError(
                "model eos_token_id does not include the tokenizer's chat-end token: "
                f"{special_tokens.im_end}"
            )

        tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if tokenizer_vocab_size > model_config.vocab_size:
            raise TokenizerError(
                "tokenizer contains more entries than the model output vocabulary: "
                f"{tokenizer_vocab_size} > {model_config.vocab_size}"
            )
        return cls(
            tokenizer,
            special_tokens=special_tokens,
            model_vocab_size=model_config.vocab_size,
        )

    @staticmethod
    def _required_token_id(tokenizer: Tokenizer, token: str) -> int:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise TokenizerError(f"required Qwen token is missing: {token!r}")
        return token_id

    @property
    def base_vocab_size(self) -> int:
        """Number of trained BPE entries, excluding added control tokens."""

        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    @property
    def vocab_size(self) -> int:
        """Number of token IDs that can be encoded and decoded."""

        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise TokenizerError(f"token is not in the tokenizer vocabulary: {token!r}")
        return token_id

    def id_to_token(self, token_id: int) -> str:
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise TokenizerError(f"token ID is not decodable: {token_id}")
        return token

    def encode(self, text: str) -> list[int]:
        """Encode text exactly as supplied, without automatically inserting tokens."""

        if not isinstance(text, str):
            raise TokenizerError(f"text must be str, got {type(text).__name__}")
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        """Decode IDs, preserving control tokens unless explicitly requested otherwise."""

        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )


def format_qwen3_chat(
    messages: Sequence[ChatMessage],
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Format a basic system/user/assistant conversation for Qwen3.

    This intentionally implements only textual chat turns.  Tool calls,
    multimodal content, and reasoning-history reconstruction are deferred.
    """

    validate_chat_messages(messages, add_generation_prompt=add_generation_prompt)
    tokens = QWEN3_CHAT_TOKENS
    prompt = "".join(
        f"{tokens.im_start}{message.role}\n{message.content}{tokens.im_end}\n"
        for message in messages
    )
    if add_generation_prompt:
        prompt += f"{tokens.im_start}assistant\n"
        if not enable_thinking:
            prompt += f"{tokens.think_start}\n\n{tokens.think_end}\n\n"
    return prompt


def validate_chat_messages(
    messages: Sequence[ChatMessage], *, add_generation_prompt: bool
) -> None:
    """Validate the role ordering supported by the minimal chat runtime."""

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
                raise TokenizerError("a system message is only allowed as the first turn")
            continue
        if message.role not in ("user", "assistant"):
            raise TokenizerError(f"unsupported chat role at message {index}: {message.role!r}")
        if message.role != expected_role:
            raise TokenizerError(
                f"expected {expected_role!r} at message {index}, got {message.role!r}"
            )
        expected_role = "assistant" if message.role == "user" else "user"

    if add_generation_prompt and messages[-1].role != "user":
        raise TokenizerError(
            "the final message must have role 'user' when adding a generation prompt"
        )
