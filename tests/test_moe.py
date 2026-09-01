from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from mini_llm.nn import GraniteMoeBlock, TopKRouter


def _explicit_moe_reference(
    block: GraniteMoeBlock, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = inputs.reshape(-1, block.hidden_size)
    logits = F.linear(flattened.float(), block.router.layer.weight.float())
    top_logits, expert_indices = torch.topk(logits, block.top_k, dim=-1)
    expert_weights = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
    output = torch.zeros_like(flattened)

    for token_index in range(flattened.shape[0]):
        for slot in range(block.top_k):
            expert_index = int(expert_indices[token_index, slot])
            gate_and_up = F.linear(
                flattened[token_index], block.input_linear.weight[expert_index]
            )
            gate, up = gate_and_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_output = F.linear(
                hidden, block.output_linear.weight[expert_index]
            )
            output[token_index] += (
                expert_output
                * expert_weights[token_index, slot].to(expert_output.dtype)
            )

    return (
        output.view_as(inputs),
        logits,
        expert_indices,
        expert_weights,
    )


class TopKRouterTests(unittest.TestCase):
    def test_selects_and_normalizes_highest_scoring_experts(self) -> None:
        router = TopKRouter(hidden_size=2, num_experts=3, top_k=2)
        with torch.no_grad():
            router.layer.weight.copy_(
                torch.tensor([[2.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
            )
        inputs = torch.tensor([[2.0, 1.0], [-1.0, 3.0]])

        routing = router(inputs)

        expected_logits = torch.tensor([[4.0, 1.0, -3.0], [-2.0, 3.0, -2.0]])
        torch.testing.assert_close(routing.logits, expected_logits)
        self.assertEqual(routing.expert_indices.tolist(), [[0, 1], [1, 0]])
        torch.testing.assert_close(
            routing.expert_weights.sum(dim=-1), torch.ones(2)
        )
        self.assertEqual(routing.logits.dtype, torch.float32)
        self.assertEqual(routing.expert_weights.dtype, torch.float32)

    def test_router_results_are_fp32_for_reduced_precision_inputs(self) -> None:
        torch.manual_seed(37)
        router = TopKRouter(hidden_size=2, num_experts=3, top_k=2).to(
            dtype=torch.bfloat16
        )
        inputs = torch.randn(2, 2, dtype=torch.bfloat16)

        routing = router(inputs)
        expected = F.linear(inputs.float(), router.layer.weight.float())

        self.assertEqual(routing.logits.dtype, torch.float32)
        self.assertEqual(routing.expert_weights.dtype, torch.float32)
        torch.testing.assert_close(routing.logits, expected, rtol=0.0, atol=0.0)


class GraniteMoeBlockTests(unittest.TestCase):
    def test_supports_meta_construction_for_checkpoint_loading(self) -> None:
        with torch.device("meta"):
            block = GraniteMoeBlock(
                hidden_size=4,
                intermediate_size=3,
                num_experts=5,
                top_k=2,
            )

        self.assertTrue(all(parameter.is_meta for parameter in block.parameters()))

    def test_parameter_names_and_shapes_match_packed_checkpoint(self) -> None:
        block = GraniteMoeBlock(
            hidden_size=8,
            intermediate_size=4,
            num_experts=32,
            top_k=8,
        )

        self.assertEqual(block.router.layer.weight.shape, (32, 8))
        self.assertEqual(block.input_linear.weight.shape, (32, 8, 8))
        self.assertEqual(block.output_linear.weight.shape, (32, 8, 4))
        self.assertEqual(
            set(block.state_dict()),
            {
                "input_linear.weight",
                "output_linear.weight",
                "router.layer.weight",
            },
        )

    def test_matches_explicit_token_and_expert_loops(self) -> None:
        torch.manual_seed(41)
        block = GraniteMoeBlock(
            hidden_size=4,
            intermediate_size=3,
            num_experts=5,
            top_k=2,
        )
        inputs = torch.randn(2, 3, 4)

        expected, logits, indices, weights = _explicit_moe_reference(block, inputs)
        actual, routing = block(inputs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(routing.logits, logits)
        self.assertTrue(torch.equal(routing.expert_indices, indices))
        torch.testing.assert_close(routing.expert_weights, weights)

    def test_single_token_batched_experts_match_explicit_reference(self) -> None:
        torch.manual_seed(67)
        block = GraniteMoeBlock(
            hidden_size=4,
            intermediate_size=3,
            num_experts=5,
            top_k=2,
        )
        inputs = torch.randn(1, 1, 4)
        expected, logits, indices, weights = _explicit_moe_reference(block, inputs)

        # Exercise the MPS-oriented helper directly on CPU so its equation is
        # covered even when the test environment has no Apple GPU.
        with patch.object(
            block.input_linear,
            "forward_expert",
            side_effect=AssertionError("single-token path used expert loop"),
        ):
            flattened = inputs.reshape(-1, block.hidden_size)
            routing = block.router(flattened)
            actual, routing = block._forward_single_token(
                inputs, flattened, routing
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(routing.logits, logits)
        self.assertTrue(torch.equal(routing.expert_indices, indices))
        torch.testing.assert_close(routing.expert_weights, weights)

    def test_multiple_token_batched_experts_match_explicit_reference(self) -> None:
        torch.manual_seed(71)
        block = GraniteMoeBlock(
            hidden_size=4,
            intermediate_size=3,
            num_experts=5,
            top_k=2,
        ).to(dtype=torch.float16)
        inputs = torch.randn(2, 3, 4, dtype=torch.float16)
        expected, logits, indices, weights = _explicit_moe_reference(block, inputs)
        flattened = inputs.reshape(-1, block.hidden_size)
        routing = block.router(flattened)

        with patch.object(
            block.input_linear,
            "forward_expert",
            side_effect=AssertionError("padded path used expert loop"),
        ):
            actual, routing = block._forward_multiple_tokens(
                inputs, flattened, routing
            )

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(routing.logits, logits)
        self.assertTrue(torch.equal(routing.expert_indices, indices))
        torch.testing.assert_close(routing.expert_weights, weights)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_dispatch_uses_batched_expert_paths(self) -> None:
        torch.manual_seed(79)
        block = GraniteMoeBlock(
            hidden_size=4,
            intermediate_size=3,
            num_experts=5,
            top_k=2,
        ).to(device="cuda", dtype=torch.float16)

        for sequence_length in (1, 3):
            with self.subTest(sequence_length=sequence_length):
                inputs = torch.randn(
                    1, sequence_length, 4, device="cuda", dtype=torch.float16
                )
                expected, logits, indices, weights = _explicit_moe_reference(
                    block, inputs
                )
                with patch.object(
                    block.input_linear,
                    "forward_expert",
                    side_effect=AssertionError("CUDA used the CPU expert loop"),
                ):
                    actual, routing = block(inputs)

                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
                torch.testing.assert_close(routing.logits, logits)
                self.assertTrue(torch.equal(routing.expert_indices, indices))
                torch.testing.assert_close(routing.expert_weights, weights)

    def test_inactive_expert_weights_do_not_affect_output(self) -> None:
        block = GraniteMoeBlock(
            hidden_size=1,
            intermediate_size=1,
            num_experts=2,
            top_k=1,
        )
        with torch.no_grad():
            block.router.layer.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            block.input_linear.weight[0].fill_(1.0)
            block.output_linear.weight[0].fill_(1.0)
            block.input_linear.weight[1].fill_(float("nan"))
            block.output_linear.weight[1].fill_(float("nan"))

        output, routing = block(torch.ones(1, 2, 1))

        self.assertEqual(routing.expert_indices.tolist(), [[0], [0]])
        self.assertTrue(torch.isfinite(output).all())

    def test_preserves_input_shape_and_execution_dtype(self) -> None:
        block = GraniteMoeBlock(
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
        ).to(dtype=torch.bfloat16)
        inputs = torch.randn(2, 3, 4, dtype=torch.bfloat16)

        output, _ = block(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(output.dtype, inputs.dtype)
        self.assertTrue(torch.isfinite(output).all())

    def test_rejects_invalid_expert_count_and_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            GraniteMoeBlock(
                hidden_size=4,
                intermediate_size=2,
                num_experts=4,
                top_k=5,
            )


if __name__ == "__main__":
    unittest.main()
