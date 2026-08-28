"""High-level runtime with explicit architecture and device selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterator, Sequence

import torch

from mini_llm.config import GraniteMoeConfig, load_config
from mini_llm.generation import GenerationEvent, generate as generate_text
from mini_llm.interfaces import ChatMessage, RuntimeCausalLM, RuntimeTokenizer
from mini_llm.qwen_model import Qwen3ForCausalLM
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import Qwen3Tokenizer


class EngineError(ValueError):
    """Raised when an execution device, dtype, or context is unsupported."""


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to MPS when available, otherwise CPU."""

    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise EngineError(f"invalid device {requested!r}") from exc
    if device.type not in ("cpu", "mps"):
        raise EngineError(
            f"unsupported device {device}; this runtime supports cpu and mps"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EngineError(
            "MPS was requested but is unavailable; use device='cpu' or check "
            "this PyTorch build and macOS environment"
        )
    return device


def resolve_dtype(
    requested: str | torch.dtype = "auto",
    *,
    device: torch.device,
) -> torch.dtype:
    """Resolve automatic precision: FP16 on MPS and FP32 on CPU."""

    if requested == "auto":
        # Prefer FP16 rather than BF16 on MPS intentionally. BF16 spends more
        # bits on exponent range but has only 7 significand bits versus FP16's
        # 10. Qwen's normalized inference activations usually do not need the
        # extra range, while the coarser BF16 rounding can accumulate through
        # attention, residuals, and MLPs and eventually change greedy tokens.
        # MPS attention kernels may also differ numerically from CPU kernels.
        # Keep BF16 available as an explicit override, but not the default.
        return torch.float16 if device.type == "mps" else torch.float32
    if isinstance(requested, torch.dtype):
        dtype = requested
    elif isinstance(requested, str) and requested in _DTYPES:
        dtype = _DTYPES[requested]
    else:
        supported = ", ".join(_DTYPES)
        raise EngineError(
            f"unsupported dtype {requested!r}; expected auto, {supported}, "
            "or a floating-point torch.dtype"
        )
    if not dtype.is_floating_point:
        raise EngineError(f"model dtype must be floating point, got {dtype}")
    return dtype


def synchronize_device(device: torch.device) -> None:
    """Wait for asynchronous work on an accelerator to finish."""

    if device.type == "mps":
        torch.mps.synchronize()


@dataclass(slots=True)
class Engine:
    """Loaded model, tokenizer, placement policy, and one-request runtime."""

    model: RuntimeCausalLM
    tokenizer: RuntimeTokenizer
    device: torch.device
    dtype: torch.dtype
    max_seq_len: int
    load_seconds: float

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        max_seq_len: int = 4096,
    ) -> "Engine":
        """Select, load, and place one supported checkpoint for inference."""

        selected_device = resolve_device(device)
        selected_dtype = resolve_dtype(dtype, device=selected_device)
        started = time.perf_counter()
        config = load_config(model_dir)
        if isinstance(config, GraniteMoeConfig):
            raise EngineError(
                "Granite MoE configuration is recognized, but its model and "
                "tokenizer are implemented in later components"
            )
        tokenizer = Qwen3Tokenizer.from_model_dir(model_dir, model_config=config)
        model = Qwen3ForCausalLM.from_model_dir(model_dir)
        model.config.validate_context_length(max_seq_len)
        model.requires_grad_(False)
        model.to(device=selected_device, dtype=selected_dtype)
        # Model precision applies to learned weights and activations, but RoPE
        # angles should still start from FP32 inverse frequencies. ``Module.to``
        # converts every floating buffer, so reconstruct this derived buffer on
        # the selected device after placement.
        model.materialize_derived_buffers(selected_device)
        synchronize_device(selected_device)
        load_seconds = time.perf_counter() - started
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=selected_device,
            dtype=selected_dtype,
            max_seq_len=max_seq_len,
            load_seconds=load_seconds,
        )

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int,
        sampling: SamplingConfig = SamplingConfig(),
        enable_thinking: bool = False,
    ) -> Iterator[GenerationEvent]:
        """Format complete chat history and stream generated text."""

        return generate_text(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            enable_thinking=enable_thinking,
            max_seq_len=self.max_seq_len,
            synchronize=self.synchronize,
        )

    def synchronize(self) -> None:
        """Wait for this engine's device to finish queued work."""

        synchronize_device(self.device)
