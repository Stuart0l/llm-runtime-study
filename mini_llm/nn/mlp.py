"""SwiGLU feed-forward network used by each Qwen3 decoder layer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

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

        gated = F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.down_proj(gated)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
