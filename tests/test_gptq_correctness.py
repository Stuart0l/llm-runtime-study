from __future__ import annotations

from pathlib import Path
import unittest

from torch.nn import functional as F
import torch

from mini_llm.checkpoint import SafeTensorCheckpoint
from mini_llm.quantization import GPTQMarlinLinear
from mini_llm.quantization.fusion import FusedMarlinProjection


MODEL_DIRS = (
    (Path(__file__).parents[1] / "models" / "qwen3-0.6b-int4", 4),
    (Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8", 8),
)
INT8_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"


def _dequantize_gptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Return the logical [in_features, out_features] GPTQ weight matrix."""

    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    packed_weights = (qweight.unsqueeze(1) >> shifts[None, :, None]) & mask
    packed_zeros = (qzeros.unsqueeze(-1) >> shifts) & mask
    in_features = qweight.shape[0] * pack_factor
    out_features = qweight.shape[1]
    zeros = packed_zeros.reshape(-1, out_features) + 1
    weights = packed_weights.reshape(in_features, out_features)
    return (
        (weights.reshape(zeros.shape[0], -1, out_features) - zeros[:, None, :])
        .to(scales.dtype)
        .mul(scales[:, None, :])
        .reshape(in_features, out_features)
    )


class GPTQReferenceTests(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and all(path.is_dir() for path, _ in MODEL_DIRS),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_fused_marlin_matches_explicit_dequantization(self) -> None:
        projections = (
            ("model.layers.0.self_attn.q_proj", 1024, 2048),
            ("model.layers.0.mlp.down_proj", 3072, 1024),
        )

        for model_dir, bits in MODEL_DIRS:
            checkpoint = SafeTensorCheckpoint.from_model_dir(model_dir)
            for prefix, in_features, out_features in projections:
                tensors = checkpoint.get_tensors(
                    f"{prefix}.{suffix}"
                    for suffix in ("qweight", "qzeros", "scales", "g_idx")
                )
                layer = GPTQMarlinLinear(
                    in_features, out_features, bits=bits
                )
                layer.load_state_dict(
                    {
                        suffix: tensors[f"{prefix}.{suffix}"]
                        for suffix in ("qweight", "qzeros", "scales", "g_idx")
                    },
                    strict=True,
                    assign=True,
                )
                inputs = torch.randn(
                    3, in_features, dtype=torch.float16, device="cuda"
                )
                layer.prepare("cuda")

                actual = layer(inputs)
                reference_weight = _dequantize_gptq(
                    tensors[f"{prefix}.qweight"],
                    tensors[f"{prefix}.qzeros"],
                    tensors[f"{prefix}.scales"],
                    bits,
                ).T.to("cuda")
                expected = F.linear(inputs, reference_weight)

                torch.testing.assert_close(
                    actual, expected, rtol=0.01, atol=0.01
                )

    @unittest.skipUnless(
        torch.cuda.is_available() and INT8_MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_fused_qkv_and_gate_up_match_individual_marlin_projections(self) -> None:
        checkpoint = SafeTensorCheckpoint.from_model_dir(INT8_MODEL_DIR)
        groups = (
            (
                (
                    "model.layers.0.self_attn.q_proj",
                    "model.layers.0.self_attn.k_proj",
                    "model.layers.0.self_attn.v_proj",
                ),
                (2048, 1024, 1024),
            ),
            (
                (
                    "model.layers.0.mlp.gate_proj",
                    "model.layers.0.mlp.up_proj",
                ),
                (3072, 3072),
            ),
        )

        for prefixes, widths in groups:
            parts = []
            for prefix, width in zip(prefixes, widths, strict=True):
                tensors = checkpoint.get_tensors(
                    f"{prefix}.{suffix}"
                    for suffix in ("qweight", "qzeros", "scales", "g_idx")
                )
                layer = GPTQMarlinLinear(1024, width, bits=8)
                layer.load_state_dict(
                    {
                        suffix: tensors[f"{prefix}.{suffix}"]
                        for suffix in ("qweight", "qzeros", "scales", "g_idx")
                    },
                    strict=True,
                    assign=True,
                )
                layer.prepare("cuda")
                parts.append(layer)

            fused = FusedMarlinProjection(*parts)
            fused.prepare("cuda")
            inputs = torch.randn(2, 1024, dtype=torch.float16, device="cuda")

            expected = torch.cat([part(inputs) for part in parts], dim=-1)
            actual = fused(inputs)

            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
