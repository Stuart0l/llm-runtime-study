"""Derived runtime fusion for compatible GPTQ-Marlin projections."""

from __future__ import annotations

import torch

from mini_llm.quantization import GPTQMarlinLinear


class FusedMarlinProjection:
    """One transient Marlin layout made by concatenating output projections.

    Component modules retain the checkpoint-owned GPTQ tensors.  The combined
    module is deliberately an ordinary attribute of this helper rather than a
    child of the model, so its repacked CUDA tensors never appear in a state
    dictionary.
    """

    def __init__(self, *components: GPTQMarlinLinear) -> None:
        if len(components) < 2:
            raise ValueError("a fused Marlin projection needs at least two components")
        first = components[0]
        if any(component.in_features != first.in_features for component in components):
            raise ValueError("fused Marlin components must share in_features")
        if any(component.bits != first.bits for component in components):
            raise ValueError("fused Marlin components must share bit width")
        self.components = components
        self.in_features = first.in_features
        self.out_features = sum(component.out_features for component in components)
        self._backend: GPTQMarlinLinear | None = None

    @staticmethod
    def _canonical(
        component: GPTQMarlinLinear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qweight = component._canonical_qweight
        scales = component._canonical_scales
        if qweight is None:
            qweight = component.qweight.detach().cpu().contiguous()
            component._canonical_qweight = qweight
        if scales is None:
            scales = component.scales.detach().cpu().contiguous()
            component._canonical_scales = scales
        return qweight, scales

    def prepare(self, device: torch.device | str) -> None:
        canonical = [self._canonical(component) for component in self.components]
        combined = GPTQMarlinLinear(
            self.in_features, self.out_features, bits=self.components[0].bits
        )
        groups = self.in_features // 128
        pack_factor = self.components[0].pack_factor
        combined.load_state_dict(
            {
                "qweight": torch.cat(
                    [qweight for qweight, _ in canonical], dim=1
                ),
                "qzeros": torch.empty(
                    groups,
                    self.out_features // pack_factor,
                    dtype=torch.int32,
                ),
                "scales": torch.cat([scales for _, scales in canonical], dim=1),
                "g_idx": torch.arange(self.in_features, dtype=torch.int32).div(
                    128, rounding_mode="floor"
                ),
            },
            assign=True,
        )
        combined.prepare(device)
        # Component tensors are the canonical source for later device moves.
        # Do not retain a second combined CPU copy.
        combined._canonical_qweight = None
        combined._canonical_scales = None
        self._backend = combined

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("call prepare(device) before fused Marlin forward")
        return self._backend(inputs)

