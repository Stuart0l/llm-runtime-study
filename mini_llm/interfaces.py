"""Small runtime contracts shared by supported model architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import torch

from mini_llm.config import DecoderConfig


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One textual message accepted by the generation runtime."""

    role: Literal["system", "user", "assistant"]
    content: str


class RuntimeTokenizer(Protocol):
    """Tokenizer operations required by architecture-neutral generation."""

    def encode(self, text: str) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...

    def format_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_thinking: bool = False,
    ) -> str: ...


class RuntimeCausalLM(Protocol):
    """Model operations required by placement and token generation."""

    config: DecoderConfig

    @property
    def input_device(self) -> torch.device: ...

    def setup_cache(self, capacity: int) -> None: ...

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def requires_grad_(self, requires_grad: bool = True) -> "RuntimeCausalLM": ...

    def to(self, *args: object, **kwargs: object) -> "RuntimeCausalLM": ...

    def materialize_derived_buffers(self, device: torch.device) -> None: ...
