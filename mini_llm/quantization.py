"""GPTQ projections backed by vLLM's compiled Marlin operators."""

from __future__ import annotations

from functools import cache
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import platform

import torch
from torch import nn


_VLLM_VERSION = "0.28.0"
_GPTQ_TYPE_IDS = {
    4: (4 << 8) | (8 << 17) | (1 << 50),
    8: (8 << 8) | (128 << 17) | (1 << 50),
}
_SCALE_PERMUTATION = tuple(i + 8 * j for i in range(8) for j in range(8))


@cache
def _load_vllm_marlin_ops() -> None:
    """Load vLLM's extension directly, without importing its Python package."""

    try:
        installed = distribution("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "gptq-marlin requires the vLLM binary wheel; install "
            f"it with `pip install --no-deps vllm=={_VLLM_VERSION}`"
        ) from exc
    if installed.version != _VLLM_VERSION:
        raise RuntimeError(
            f"gptq-marlin requires vllm=={_VLLM_VERSION}, got {installed.version}"
        )

    libraries = [
        Path(installed.locate_file(file))
        for file in installed.files or ()
        if str(file).startswith("vllm/_C_stable_libtorch")
        and str(file).endswith(".so")
    ]
    if len(libraries) != 1:
        raise RuntimeError(
            "gptq-marlin could not locate vLLM's stable libtorch extension"
        )
    try:
        torch.ops.load_library(str(libraries[0]))
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"gptq-marlin could not load {libraries[0]}: {exc}") from exc

    missing = [
        name
        for name in ("gptq_marlin_repack", "marlin_gemm")
        if not hasattr(torch.ops._C, name)
    ]
    if missing:
        raise RuntimeError(
            "vLLM did not register required gptq-marlin operators: "
            + ", ".join(missing)
        )


def validate_gptq_marlin_device(device: torch.device | str) -> torch.device:
    """Validate the CUDA target and initialize the pinned Marlin backend."""

    target = torch.device(device)
    if platform.system() != "Linux":
        raise RuntimeError("gptq-marlin requires Linux")
    if target.type != "cuda":
        raise RuntimeError(f"gptq-marlin requires a CUDA device, got {target}")
    if not torch.cuda.is_available():
        raise RuntimeError("gptq-marlin requires an available CUDA device")
    if torch.cuda.get_device_capability(target) < (7, 5):
        raise RuntimeError("gptq-marlin requires CUDA compute capability 7.5+")
    _load_vllm_marlin_ops()
    return target


class GPTQMarlinLinear(nn.Module):
    """Group-size-128 symmetric GPTQ linear projection for CUDA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int,
        bias: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if bits not in _GPTQ_TYPE_IDS:
            raise ValueError(f"gptq-marlin supports 4 or 8 bits, got {bits}")
        if not (
            (out_features % 64 == 0 and in_features % 128 == 0)
            or (out_features % 128 == 0 and in_features % 64 == 0)
        ):
            raise ValueError(
                "gptq-marlin requires a 64x128-aligned projection, got "
                f"in_features={in_features}, out_features={out_features}"
            )
        if bias:
            raise NotImplementedError("Stage 1 gptq-marlin does not support bias")

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.pack_factor = 32 // bits
        self.qweight = nn.Parameter(
            torch.empty(
                in_features // self.pack_factor,
                out_features,
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )
        self.qzeros = nn.Parameter(
            torch.empty(
                in_features // 128,
                out_features // self.pack_factor,
                dtype=torch.int32,
                device=device,
            ),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty(in_features // 128, out_features, dtype=torch.float16, device=device),
            requires_grad=False,
        )
        self.g_idx = nn.Parameter(
            torch.empty(in_features, dtype=torch.int32, device=device),
            requires_grad=False,
        )
        self.register_buffer(
            "workspace", torch.empty(0, dtype=torch.int32, device=device), persistent=False
        )
        self._canonical_qweight: torch.Tensor | None = None
        self._canonical_scales: torch.Tensor | None = None

    def _apply(self, fn, recurse: bool = True):
        # Canonical GPTQ tensors stay on CPU; prepare() owns the CUDA layout.
        return self

    def prepare(self, device: torch.device | str) -> None:
        """Preserve canonical CPU tensors and build the Marlin CUDA layout."""

        target = validate_gptq_marlin_device(device)
        if any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError("cannot prepare an unmaterialized gptq-marlin projection")

        if self._canonical_qweight is None:
            self._canonical_qweight = self.qweight.detach().cpu().contiguous()
            self._canonical_scales = self.scales.detach().cpu().contiguous()
        assert self._canonical_qweight is not None
        assert self._canonical_scales is not None

        packed = self._canonical_qweight.to(target)
        empty_permutation = torch.empty(0, dtype=torch.int32, device=target)
        packed = torch.ops._C.gptq_marlin_repack(
            packed,
            empty_permutation,
            self.in_features,
            self.out_features,
            self.bits,
            False,
        )
        permutation = torch.tensor(
            _SCALE_PERMUTATION, dtype=torch.long, device=target
        )
        scales = self._canonical_scales.to(target)
        scales = scales.reshape(-1, 64).index_select(1, permutation)
        scales = scales.reshape(-1, self.out_features).contiguous()

        self.qweight = nn.Parameter(packed, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        self.qzeros = nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=target), requires_grad=False
        )
        self.g_idx = nn.Parameter(
            torch.empty(0, dtype=torch.int32, device=target), requires_grad=False
        )
        multiprocessors = torch.cuda.get_device_properties(target).multi_processor_count
        self.workspace = torch.zeros(multiprocessors, dtype=torch.int32, device=target)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.workspace.numel() == 0:
            raise RuntimeError("call prepare(cuda_device) before gptq-marlin forward")
        if inputs.device != self.qweight.device or inputs.dtype != torch.float16:
            raise RuntimeError(
                "gptq-marlin input must be FP16 on the prepared CUDA device"
            )
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input final dimension {self.in_features}, "
                f"got {None if inputs.ndim == 0 else inputs.shape[-1]}"
            )
        if inputs.numel() == 0:
            return inputs.new_empty((*inputs.shape[:-1], self.out_features))

        matrix = inputs.reshape(-1, self.in_features)
        output = torch.ops._C.marlin_gemm(
            matrix,
            None,
            self.qweight,
            None,
            self.scales,
            None,
            None,
            None,
            None,
            None,
            self.workspace,
            _GPTQ_TYPE_IDS[self.bits],
            matrix.shape[0],
            self.out_features,
            self.in_features,
            True,
            False,
            True,
            False,
        )
        return output.reshape(*inputs.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size=128, bias=False"
        )
