"""SwiGLU feed-forward network used by each Qwen3 decoder layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from mini_llm.quantization.fusion import FusedMarlinProjection
from mini_llm.quantization import GPTQMarlinLinear


class SwiGLUFeedForward(nn.Module):
    r"""Apply Qwen3's gated feed-forward transformation.

    For an input vector :math:`x`, the module computes

    .. math::

        \operatorname{down}\left(
            \operatorname{SiLU}(\operatorname{gate}(x))
            \odot \operatorname{up}(x)
        \right)

    ``gate_proj`` and ``up_proj`` independently expand the hidden dimension to
    the intermediate dimension.  Their elementwise product is then projected
    back to the hidden dimension by ``down_proj``.  The attribute names match
    the Qwen3 Safetensors checkpoint exactly when this module is stored as a
    decoder layer's ``mlp`` attribute.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        linear_type: type[nn.Module] = nn.Linear,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = linear_type(hidden_size, intermediate_size, bias=False)
        self.up_proj = linear_type(hidden_size, intermediate_size, bias=False)
        self.down_proj = linear_type(intermediate_size, hidden_size, bias=False)
        self._gate_up_fusion: FusedMarlinProjection | None = None

    def prepare_gate_up_fusion(self, device: torch.device | str) -> None:
        if not all(
            isinstance(projection, GPTQMarlinLinear)
            for projection in (self.gate_proj, self.up_proj)
        ):
            raise RuntimeError("gate/up fusion requires GPTQ-Marlin projections")
        fusion = FusedMarlinProjection(self.gate_proj, self.up_proj)
        fusion.prepare(device)
        self._gate_up_fusion = fusion

    def _apply(self, fn, recurse: bool = True):
        self._gate_up_fusion = None
        return super()._apply(fn, recurse=recurse)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not inputs.is_floating_point():
            raise TypeError(
                f"SwiGLUFeedForward requires floating-point input, got {inputs.dtype}"
            )
        if inputs.ndim == 0 or inputs.shape[-1] != self.hidden_size:
            actual = None if inputs.ndim == 0 else inputs.shape[-1]
            raise ValueError(
                "SwiGLUFeedForward expected final dimension "
                f"{self.hidden_size}, got {actual}"
            )

        if self._gate_up_fusion is None:
            gate_states = self.gate_proj(inputs)
            up_states = self.up_proj(inputs)
        else:
            gate_states, up_states = self._gate_up_fusion(inputs).split(
                self.intermediate_size, dim=-1
            )
        gated = F.silu(gate_states) * up_states
        return self.down_proj(gated)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
