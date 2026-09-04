from __future__ import annotations

from pathlib import Path
import unittest

import torch

from mini_llm.checkpoint import SafeTensorCheckpoint
from mini_llm.fusion import FusedMarlinProjection
from mini_llm.quantization import GPTQMarlinLinear


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"


class FusedMarlinProjectionTests(unittest.TestCase):
    def test_rejects_insufficient_or_incompatible_components(self) -> None:
        first = GPTQMarlinLinear(128, 128)
        second = GPTQMarlinLinear(256, 128)

        with self.assertRaisesRegex(ValueError, "at least two"):
            FusedMarlinProjection(first)
        with self.assertRaisesRegex(ValueError, "share in_features"):
            FusedMarlinProjection(first, second)

    @unittest.skipUnless(
        torch.cuda.is_available() and MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_fused_qkv_and_gate_up_match_individual_marlin_projections(self) -> None:
        checkpoint = SafeTensorCheckpoint.from_model_dir(MODEL_DIR)
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
                layer = GPTQMarlinLinear(1024, width)
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
