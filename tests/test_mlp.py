from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from mini_llm.nn import SwiGLUFeedForward


class SwiGLUFeedForwardTests(unittest.TestCase):
    def test_projection_shapes_match_qwen3_checkpoint_layout(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=4, intermediate_size=6)

        self.assertEqual(mlp.gate_proj.weight.shape, (6, 4))
        self.assertEqual(mlp.up_proj.weight.shape, (6, 4))
        self.assertEqual(mlp.down_proj.weight.shape, (4, 6))
        self.assertIsNone(mlp.gate_proj.bias)
        self.assertIsNone(mlp.up_proj.bias)
        self.assertIsNone(mlp.down_proj.bias)

    def test_matches_explicit_reference_equation(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=2, intermediate_size=3)
        inputs = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(
                torch.tensor([[1.0, 0.5], [-0.5, 2.0], [1.5, -1.0]])
            )
            mlp.up_proj.weight.copy_(
                torch.tensor([[0.25, 1.0], [2.0, -0.5], [-1.0, 0.75]])
            )
            mlp.down_proj.weight.copy_(
                torch.tensor([[1.0, -0.5, 0.25], [-0.25, 0.75, 1.5]])
            )

        gate = F.linear(inputs, mlp.gate_proj.weight)
        up = F.linear(inputs, mlp.up_proj.weight)
        expected = F.linear(F.silu(gate) * up, mlp.down_proj.weight)

        torch.testing.assert_close(mlp(inputs), expected)

    def test_preserves_leading_batch_and_sequence_dimensions(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=4, intermediate_size=12)

        output = mlp(torch.randn(2, 5, 4))

        self.assertEqual(output.shape, (2, 5, 4))

    def test_gate_can_suppress_intermediate_features(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=2, intermediate_size=2)
        with torch.no_grad():
            mlp.gate_proj.weight.zero_()
            mlp.up_proj.weight.fill_(1.0)
            mlp.down_proj.weight.fill_(1.0)

        output = mlp(torch.tensor([[2.0, 3.0]]))

        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_rejects_wrong_final_dimension(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=4, intermediate_size=8)

        with self.assertRaisesRegex(ValueError, "expected final dimension 4"):
            mlp(torch.ones(2, 3))

    def test_rejects_integer_input(self) -> None:
        mlp = SwiGLUFeedForward(hidden_size=2, intermediate_size=4)

        with self.assertRaisesRegex(TypeError, "floating-point"):
            mlp(torch.ones(1, 2, dtype=torch.int64))

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden_size must be positive"):
            SwiGLUFeedForward(hidden_size=0, intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "intermediate_size must be positive"):
            SwiGLUFeedForward(hidden_size=4, intermediate_size=0)


if __name__ == "__main__":
    unittest.main()
