from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

import torch

from mini_llm.checkpoint import SafeTensorCheckpoint
from mini_llm.quantization import GPTQMarlinLinear


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"


class GPTQMarlinLinearTests(unittest.TestCase):
    def test_state_names_and_shapes_match_gptq_checkpoint(self) -> None:
        with torch.device("meta"):
            layer = GPTQMarlinLinear(1024, 2048)

        self.assertEqual(
            set(layer.state_dict()), {"qweight", "qzeros", "scales", "g_idx"}
        )
        self.assertEqual(layer.qweight.shape, (256, 2048))
        self.assertEqual(layer.qzeros.shape, (8, 512))
        self.assertEqual(layer.scales.shape, (8, 2048))
        self.assertEqual(layer.g_idx.shape, (1024,))

    def test_rejects_unsupported_shape_and_bias(self) -> None:
        with self.assertRaisesRegex(ValueError, "64x128-aligned"):
            GPTQMarlinLinear(96, 96)
        with self.assertRaisesRegex(NotImplementedError, "does not support bias"):
            GPTQMarlinLinear(128, 128, bias=True)

    def test_requires_preparation_before_forward(self) -> None:
        layer = GPTQMarlinLinear(128, 128)
        with self.assertRaisesRegex(RuntimeError, "call prepare"):
            layer(torch.zeros(1, 128, dtype=torch.float16))

    def test_import_does_not_import_vllm_or_torchvision(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import mini_llm.quantization; "
                "assert 'vllm' not in sys.modules; "
                "assert 'torchvision' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        torch.cuda.is_available() and MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_real_checkpoint_projection_runs_fused_marlin(self) -> None:
        checkpoint = SafeTensorCheckpoint.from_model_dir(MODEL_DIR)
        prefix = "model.layers.0.self_attn.q_proj"
        tensors = checkpoint.get_tensors(
            f"{prefix}.{suffix}"
            for suffix in ("qweight", "qzeros", "scales", "g_idx")
        )
        layer = GPTQMarlinLinear(1024, 2048)
        layer.load_state_dict(
            {
                suffix: tensors[f"{prefix}.{suffix}"]
                for suffix in ("qweight", "qzeros", "scales", "g_idx")
            },
            strict=True,
            assign=True,
        )

        layer.prepare("cuda")
        output = layer(torch.randn(2, 3, 1024, dtype=torch.float16, device="cuda"))

        self.assertEqual(output.shape, (2, 3, 2048))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all().item())
        assert layer._canonical_qweight is not None
        assert layer._canonical_scales is not None
        self.assertEqual(layer._canonical_qweight.device.type, "cpu")
        self.assertEqual(layer._canonical_scales.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
