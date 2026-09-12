"""Protocol required from causal language models by the runtime."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from mini_llm.cache import SequenceKVCache
from mini_llm.config import DecoderConfig


class RuntimeCausalLM(Protocol):
    """Model operations required by placement and token generation."""

    config: DecoderConfig

    @property
    def input_device(self) -> torch.device: ...

    def prefill(
        self, input_ids: torch.Tensor, *, cache: SequenceKVCache
    ) -> torch.Tensor: ...

    def decode(
        self,
        input_ids: torch.Tensor,
        *,
        caches: Sequence[SequenceKVCache],
    ) -> torch.Tensor: ...

    def requires_grad_(self, requires_grad: bool = True) -> "RuntimeCausalLM": ...

    def to(self, *args: object, **kwargs: object) -> "RuntimeCausalLM": ...

    def materialize_derived_buffers(self, device: torch.device) -> None: ...
