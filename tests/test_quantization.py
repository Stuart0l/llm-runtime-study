from __future__ import annotations

import subprocess
import sys
import unittest

import torch

from mini_llm.quantization import GPTQMarlinLinear


class GPTQMarlinLinearTests(unittest.TestCase):
    def test_rejects_unsupported_shape_and_bias(self) -> None:
        with self.assertRaisesRegex(ValueError, "64x128-aligned"):
            GPTQMarlinLinear(96, 96, bits=8)
        with self.assertRaisesRegex(NotImplementedError, "does not support bias"):
            GPTQMarlinLinear(128, 128, bits=8, bias=True)
        with self.assertRaisesRegex(ValueError, "supports 4 or 8 bits"):
            GPTQMarlinLinear(128, 128, bits=3)

    def test_requires_preparation_before_forward(self) -> None:
        layer = GPTQMarlinLinear(128, 128, bits=8)
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


if __name__ == "__main__":
    unittest.main()
