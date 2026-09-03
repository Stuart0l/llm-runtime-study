from __future__ import annotations

from pathlib import Path
import unittest

from torch.nn import functional as F
import torch

from mini_llm.checkpoint import SafeTensorCheckpoint
from mini_llm.quantization import GPTQMarlinLinear


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"
_BYTE_SHIFTS = torch.arange(4, dtype=torch.int32) * 8


def _dequantize_gptq_int8(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Return the logical [in_features, out_features] GPTQ weight matrix."""

    packed_weights = (qweight.unsqueeze(1) >> _BYTE_SHIFTS[None, :, None]) & 0xFF
    packed_zeros = (qzeros.unsqueeze(-1) >> _BYTE_SHIFTS) & 0xFF
    in_features = qweight.shape[0] * 4
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
    def test_dequantizes_all_bytes_and_the_packed_zero_convention(self) -> None:
        qweight = torch.tensor(
            [[0x03020100, 0x13121110, 0x23222120, 0x33323130]],
            dtype=torch.int32,
        )
        qzeros = torch.tensor([[0x09080706]], dtype=torch.int32)
        scales = torch.tensor([[0.5, 1.0, 2.0, 4.0]], dtype=torch.float16)

        actual = _dequantize_gptq_int8(qweight, qzeros, scales)
        expected = torch.tensor(
            [
                [-3.5, 8.0, 46.0, 152.0],
                [-3.0, 9.0, 48.0, 156.0],
                [-2.5, 10.0, 50.0, 160.0],
                [-2.0, 11.0, 52.0, 164.0],
            ],
            dtype=torch.float16,
        )

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_fused_marlin_matches_explicit_dequantization(self) -> None:
        checkpoint = SafeTensorCheckpoint.from_model_dir(MODEL_DIR)
        projections = (
            ("model.layers.0.self_attn.q_proj", 1024, 2048),
            ("model.layers.0.mlp.down_proj", 3072, 1024),
        )

        for prefix, in_features, out_features in projections:
            tensors = checkpoint.get_tensors(
                f"{prefix}.{suffix}"
                for suffix in ("qweight", "qzeros", "scales", "g_idx")
            )
            layer = GPTQMarlinLinear(in_features, out_features)
            layer.load_state_dict(
                {
                    suffix: tensors[f"{prefix}.{suffix}"]
                    for suffix in ("qweight", "qzeros", "scales", "g_idx")
                },
                strict=True,
                assign=True,
            )
            inputs = torch.randn(3, in_features, dtype=torch.float16, device="cuda")
            layer.prepare("cuda")

            actual = layer(inputs)
            reference_weight = _dequantize_gptq_int8(
                tensors[f"{prefix}.qweight"],
                tensors[f"{prefix}.qzeros"],
                tensors[f"{prefix}.scales"],
            ).T.to("cuda")
            expected = F.linear(inputs, reference_weight)

            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
