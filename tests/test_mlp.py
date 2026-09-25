from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from mini_llm.nn import SwiGLUFeedForward


class SwiGLUFeedForwardTests(unittest.TestCase):
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

    def test_rejects_invalid_inputs_and_dimensions(self) -> None:
        cases = (
            (
                "wrong final dimension",
                lambda: SwiGLUFeedForward(hidden_size=4, intermediate_size=8)(
                    torch.ones(2, 3)
                ),
                ValueError,
                "expected final dimension 4",
            ),
            (
                "integer input",
                lambda: SwiGLUFeedForward(hidden_size=2, intermediate_size=4)(
                    torch.ones(1, 2, dtype=torch.int64)
                ),
                TypeError,
                "floating-point",
            ),
            (
                "non-positive hidden_size",
                lambda: SwiGLUFeedForward(hidden_size=0, intermediate_size=4),
                ValueError,
                "hidden_size must be positive",
            ),
            (
                "non-positive intermediate_size",
                lambda: SwiGLUFeedForward(hidden_size=4, intermediate_size=0),
                ValueError,
                "intermediate_size must be positive",
            ),
        )

        for name, action, error, message in cases:
            with self.subTest(name), self.assertRaisesRegex(error, message):
                action()


if __name__ == "__main__":
    unittest.main()
