"""Tokenizer artifact loading, validation, encoding, and decoding.

The ``tokenizers`` package owns the trained BPE vocabulary and segmentation
algorithm. Architecture-specific prompt construction lives in ``mini_llm.chat``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer

from mini_llm.chat import (
    ChatTemplate,
    GraniteChatTemplate,
    Qwen3ChatTemplate,
    TokenizerError,
)
from mini_llm.config import DecoderConfig, GraniteMoeConfig, Qwen3Config, load_config
from mini_llm.interfaces import ChatMessage


def _load_tokenizer_artifacts(model_path: Path) -> tuple[Tokenizer, dict[str, object]]:
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if not tokenizer_path.is_file():
        raise TokenizerError(f"tokenizer file does not exist: {tokenizer_path}")
    if not tokenizer_config_path.is_file():
        raise TokenizerError(
            f"tokenizer configuration does not exist: {tokenizer_config_path}"
        )
    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TokenizerError(f"invalid JSON in {tokenizer_config_path}: {exc}") from exc
    if not isinstance(tokenizer_config, dict):
        raise TokenizerError(
            f"tokenizer configuration root must be an object: {tokenizer_config_path}"
        )
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as exc:
        raise TokenizerError(f"could not load tokenizer: {tokenizer_path}") from exc
    return tokenizer, tokenizer_config


class TextTokenizer(ABC):
    """Shared wrapper around a trained tokenizer vocabulary."""

    chat_template: type[ChatTemplate]

    def __init__(self, tokenizer: Tokenizer, *, model_vocab_size: int) -> None:
        self._tokenizer = tokenizer
        self.model_vocab_size = model_vocab_size

    @staticmethod
    def _required_token_id(tokenizer: Tokenizer, token: str, *, family: str) -> int:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise TokenizerError(f"required {family} token is missing: {token!r}")
        return token_id

    @staticmethod
    def _validate_vocab_size(tokenizer: Tokenizer, model_vocab_size: int) -> None:
        tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        if tokenizer_vocab_size > model_vocab_size:
            raise TokenizerError(
                "tokenizer contains more entries than the model output vocabulary: "
                f"{tokenizer_vocab_size} > {model_vocab_size}"
            )

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

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Decode IDs, preserving control tokens unless requested otherwise."""

        return self._tokenizer.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    @classmethod
    def format_chat(
        cls,
        messages: Sequence[ChatMessage],
        *,
        enable_thinking: bool = False,
    ) -> str:
        """Delegate chat policy to this architecture's template."""

        return cls.chat_template.format(
            messages, enable_thinking=enable_thinking
        )

    @classmethod
    @abstractmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_config: DecoderConfig | None = None,
    ) -> "TextTokenizer":
        """Load and validate architecture-specific tokenizer artifacts."""


@dataclass(frozen=True, slots=True)
class Qwen3SpecialTokenIds:
    """IDs of the control tokens needed by the first runtime version."""

    end_of_text: int
    im_start: int
    im_end: int
    think_start: int
    think_end: int


@dataclass(frozen=True, slots=True)
class GraniteSpecialTokenIds:
    """IDs of Granite's official control tokens."""

    end_of_text: int
    start_of_role: int
    end_of_role: int
    tool_call: int


class Qwen3Tokenizer(TextTokenizer):
    """Thin, validated wrapper around a local Qwen tokenizer."""

    chat_template = Qwen3ChatTemplate

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        special_tokens: Qwen3SpecialTokenIds,
        model_vocab_size: int,
    ) -> None:
        super().__init__(tokenizer, model_vocab_size=model_vocab_size)
        self.special_tokens = special_tokens

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_config: DecoderConfig | None = None,
    ) -> "Qwen3Tokenizer":
        model_path = Path(model_dir)
        if model_config is None:
            model_config = Qwen3Config.from_model_dir(model_path)
        if not isinstance(model_config, Qwen3Config):
            raise TokenizerError("Qwen3Tokenizer requires a Qwen3Config")
        tokenizer, tokenizer_config = _load_tokenizer_artifacts(model_path)
        chat_tokens = cls.chat_template.tokens
        if tokenizer_config.get("eos_token") != chat_tokens.im_end:
            raise TokenizerError(f"Qwen3 eos_token must be {chat_tokens.im_end!r}")
        if tokenizer_config.get("pad_token") != chat_tokens.end_of_text:
            raise TokenizerError(
                f"Qwen3 pad_token must be {chat_tokens.end_of_text!r}"
            )

        special_tokens = Qwen3SpecialTokenIds(
            end_of_text=cls._required_token_id(
                tokenizer, chat_tokens.end_of_text, family="Qwen3"
            ),
            im_start=cls._required_token_id(
                tokenizer, chat_tokens.im_start, family="Qwen3"
            ),
            im_end=cls._required_token_id(
                tokenizer, chat_tokens.im_end, family="Qwen3"
            ),
            think_start=cls._required_token_id(
                tokenizer, chat_tokens.think_start, family="Qwen3"
            ),
            think_end=cls._required_token_id(
                tokenizer, chat_tokens.think_end, family="Qwen3"
            ),
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

        cls._validate_vocab_size(tokenizer, model_config.vocab_size)
        return cls(
            tokenizer,
            special_tokens=special_tokens,
            model_vocab_size=model_config.vocab_size,
        )


class GraniteTokenizer(TextTokenizer):
    """Validated wrapper around Granite's local tokenizer artifacts."""

    chat_template = GraniteChatTemplate

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        special_tokens: GraniteSpecialTokenIds,
        model_vocab_size: int,
    ) -> None:
        super().__init__(tokenizer, model_vocab_size=model_vocab_size)
        self.special_tokens = special_tokens

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        model_config: DecoderConfig | None = None,
    ) -> "GraniteTokenizer":
        model_path = Path(model_dir)
        if model_config is None:
            model_config = GraniteMoeConfig.from_model_dir(model_path)
        if not isinstance(model_config, GraniteMoeConfig):
            raise TokenizerError("GraniteTokenizer requires a GraniteMoeConfig")
        tokenizer, tokenizer_config = _load_tokenizer_artifacts(model_path)
        chat_tokens = cls.chat_template.tokens

        for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
            if tokenizer_config.get(name) != chat_tokens.end_of_text:
                raise TokenizerError(
                    f"Granite {name} must be {chat_tokens.end_of_text!r}"
                )

        special_tokens = GraniteSpecialTokenIds(
            end_of_text=cls._required_token_id(
                tokenizer, chat_tokens.end_of_text, family="Granite"
            ),
            start_of_role=cls._required_token_id(
                tokenizer, chat_tokens.start_of_role, family="Granite"
            ),
            end_of_role=cls._required_token_id(
                tokenizer, chat_tokens.end_of_role, family="Granite"
            ),
            tool_call=cls._required_token_id(
                tokenizer, chat_tokens.tool_call, family="Granite"
            ),
        )
        if special_tokens.end_of_text != model_config.bos_token_id:
            raise TokenizerError(
                "model bos_token_id does not match Granite's end-of-text token: "
                f"{model_config.bos_token_id} != {special_tokens.end_of_text}"
            )
        if special_tokens.end_of_text not in model_config.eos_token_ids:
            raise TokenizerError(
                "model eos_token_id does not include Granite's end-of-text token: "
                f"{special_tokens.end_of_text}"
            )
        if special_tokens.end_of_text != model_config.pad_token_id:
            raise TokenizerError(
                "model pad_token_id does not match Granite's end-of-text token: "
                f"{model_config.pad_token_id} != {special_tokens.end_of_text}"
            )

        cls._validate_vocab_size(tokenizer, model_config.vocab_size)
        return cls(
            tokenizer,
            special_tokens=special_tokens,
            model_vocab_size=model_config.vocab_size,
        )


TOKENIZER_TYPES: dict[str, type[TextTokenizer]] = {
    "qwen3": Qwen3Tokenizer,
    "granitemoe": GraniteTokenizer,
}


def load_tokenizer(model_dir: str | Path) -> TextTokenizer:
    """Load the registered tokenizer for a supported model configuration."""

    config = load_config(model_dir)
    try:
        tokenizer_type = TOKENIZER_TYPES[config.model_type]
    except KeyError as exc:
        raise TokenizerError(
            f"no tokenizer is registered for model_type {config.model_type!r}"
        ) from exc
    return tokenizer_type.from_model_dir(model_dir, model_config=config)
