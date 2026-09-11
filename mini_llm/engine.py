"""High-level runtime with explicit architecture and device selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Iterator, Literal, Sequence

import torch

from mini_llm.cache import KVCacheManager, SequenceKVCache
from mini_llm.cache.dense import DenseKVCacheManager
from mini_llm.cache.paged import PagedKVCachePool, SequenceCacheHandle
from mini_llm.config import DecoderConfig, load_config
from mini_llm.cuda_graph import PagedDecodeGraph
from mini_llm.generation import GenerationEvent, generate as generate_text
from mini_llm.model.base import CausalLMBase
from mini_llm.model.contracts import RuntimeCausalLM
from mini_llm.model.loader import load_model
from mini_llm.quantization import validate_gptq_marlin_device
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage, RuntimeTokenizer, load_tokenizer


class EngineError(ValueError):
    """Raised when an execution device, dtype, or context is unsupported."""


CacheBackend = Literal["paged", "dense"]


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def infer_quantization(model_config: DecoderConfig) -> str:
    """Infer the only valid runtime mode from checkpoint metadata."""

    return (
        "gptq-marlin"
        if model_config.quantization_config is not None
        else "dense"
    )


def _validate_quantized_placement(
    quantization: str, device: torch.device, dtype: torch.dtype
) -> None:
    if quantization != "gptq-marlin":
        return
    if device.type != "cuda":
        raise EngineError(f"gptq-marlin requires CUDA, got {device}")
    if dtype != torch.float16:
        raise EngineError(f"gptq-marlin requires float16, got {dtype}")
    try:
        validate_gptq_marlin_device(device)
    except RuntimeError as exc:
        raise EngineError(str(exc)) from exc


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve and validate one supported PyTorch execution device."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise EngineError(f"invalid device {requested!r}") from exc
    if device.type not in ("cpu", "mps", "cuda"):
        raise EngineError(
            f"unsupported device {device}; this runtime supports cpu, mps, and cuda"
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise EngineError(
            "MPS was requested but is unavailable; use device='cpu' or check "
            "this PyTorch build and macOS environment"
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise EngineError(
                "CUDA was requested but is unavailable; use device='cpu' or check "
                "the NVIDIA driver and CUDA-enabled PyTorch installation"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise EngineError(
                f"CUDA device index {device.index} is unavailable; found "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
    return device


def resolve_dtype(
    requested: str | torch.dtype = "auto",
    *,
    device: torch.device,
) -> torch.dtype:
    """Resolve automatic precision: FP16 on accelerators and FP32 on CPU."""

    if requested == "auto":
        # FP16 reduces accelerator weight/cache memory and uses CUDA Tensor
        # Cores. It is also intentional on MPS: BF16 has only 7 significand
        # bits versus FP16's 10, and its coarser rounding can accumulate through
        # attention, residuals, and feed-forward blocks. Keep BF16 available as
        # an explicit override, but not the default.
        return torch.float16 if device.type in ("mps", "cuda") else torch.float32
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
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(slots=True)
class Engine:
    """Loaded model, tokenizer, placement policy, and one-request runtime."""

    model: RuntimeCausalLM
    tokenizer: RuntimeTokenizer
    device: torch.device
    dtype: torch.dtype
    max_seq_len: int
    load_seconds: float
    quantization: str = "dense"
    cache_backend: CacheBackend = "paged"
    use_cuda_graph: bool = True
    _cache_manager: KVCacheManager | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.cache_backend not in ("paged", "dense"):
            raise EngineError(
                f"unsupported cache backend {self.cache_backend!r}; "
                "expected 'paged' or 'dense'"
            )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
        max_seq_len: int = 4096,
        cache_backend: CacheBackend = "paged",
        use_cuda_graph: bool = True,
    ) -> "Engine":
        """Select, load, and place one supported checkpoint for inference."""

        started = time.perf_counter()
        config = load_config(model_dir)
        selected_quantization = infer_quantization(config)
        selected_device = resolve_device(device)
        selected_dtype = resolve_dtype(dtype, device=selected_device)
        _validate_quantized_placement(
            selected_quantization, selected_device, selected_dtype
        )
        tokenizer = load_tokenizer(model_dir, model_config=config)
        model = load_model(model_dir, model_config=config)
        model.config.validate_context_length(max_seq_len)
        model.requires_grad_(False)
        loaded_parameter = next(model.parameters())
        engine = cls(
            model=model,
            tokenizer=tokenizer,
            device=model.input_device,
            dtype=loaded_parameter.dtype,
            max_seq_len=max_seq_len,
            load_seconds=0.0,
            quantization=selected_quantization,
            cache_backend=cache_backend,
            use_cuda_graph=use_cuda_graph,
        )
        engine.to(device=selected_device, dtype=selected_dtype)
        load_seconds = time.perf_counter() - started
        engine.load_seconds = load_seconds
        return engine

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
            cache_manager=self._ensure_cache_manager(),
            decode_factory=self._make_decode,
        )

    def _make_decode(
        self, cache: SequenceKVCache
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if (
            self.use_cuda_graph
            and self.device.type == "cuda"
            and isinstance(cache, SequenceCacheHandle)
            and isinstance(self.model, CausalLMBase)
        ):
            return PagedDecodeGraph(self.model, cache).replay
        return lambda input_ids: self.model.decode(input_ids, cache=cache)

    def _ensure_cache_manager(self) -> KVCacheManager:
        if self._cache_manager is None:
            manager_type = (
                PagedKVCachePool
                if self.cache_backend == "paged"
                else DenseKVCacheManager
            )
            self._cache_manager = manager_type(
                self.model.config,
                self.max_seq_len,
                dtype=self.dtype,
                device=self.device,
            )
        return self._cache_manager

    @property
    def cache_manager(self) -> KVCacheManager:
        return self._ensure_cache_manager()

    @property
    def last_cache_num_bytes(self) -> int:
        return self._ensure_cache_manager().last_allocation_num_bytes

    @property
    def last_cache_capacity(self) -> int:
        return self._ensure_cache_manager().last_allocation_capacity

    def allocate_cache(self, capacity: int) -> SequenceKVCache:
        return self._ensure_cache_manager().allocate(capacity)

    def release_cache(self, cache: SequenceKVCache) -> None:
        self._ensure_cache_manager().release(cache)

    def to(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "Engine":
        """Move the resident model without loading its checkpoint again.

        As with ``torch.nn.Module.to``, dtype conversion changes the resident
        weights. Widening a model after a lossy downcast does not restore the
        checkpoint's original precision.
        """

        selected_device = (
            self.device if device is None else resolve_device(device)
        )
        selected_dtype = (
            self.dtype
            if dtype is None
            else resolve_dtype(dtype, device=selected_device)
        )
        if selected_device == self.device and selected_dtype == self.dtype:
            return self
        _validate_quantized_placement(
            self.quantization, selected_device, selected_dtype
        )
        if self._cache_manager is not None and self._cache_manager.active_sequences:
            raise EngineError("cannot move an engine while request caches are active")

        # Module.to preserves eval mode and requires_grad flags. Cache managers
        # are runtime-owned and rebuilt lazily for the new placement.
        self.model.to(device=selected_device, dtype=selected_dtype)
        if self.quantization == "gptq-marlin":
            self.model.prepare_quantized(selected_device)
        # Model precision applies to learned weights and activations, but RoPE
        # angles should still start from FP32 inverse frequencies. Module.to
        # converts every floating buffer, so rebuild this buffer afterward.
        self.model.materialize_derived_buffers(selected_device)
        synchronize_device(selected_device)
        self.device = selected_device
        self.dtype = selected_dtype
        self._cache_manager = None
        return self

    def synchronize(self) -> None:
        """Wait for this engine's device to finish queued work."""

        synchronize_device(self.device)
