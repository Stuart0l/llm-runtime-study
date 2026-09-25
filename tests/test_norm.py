from __future__ import annotations

import unittest

import torch

from mini_llm.nn import RMSNorm, normalize_qwen3_queries_and_keys


class RMSNormTests(unittest.TestCase):
    def test_matches_explicit_reference_equation(self) -> None:
        inputs = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, 3.0]],
            dtype=torch.float32,
        )
        norm = RMSNorm(4, eps=1e-6)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        expected = (
            inputs
            * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-6)
            * norm.weight
        )

        actual = norm(inputs)

        torch.testing.assert_close(actual, expected)

    def test_fp16_statistics_accumulate_in_fp32_without_overflow(self) -> None:
        inputs = torch.tensor([[300.0, 400.0]], dtype=torch.float16)
        norm = RMSNorm(2).to(dtype=torch.float16)

        output = norm(inputs)

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.dtype, torch.float16)
        expected = torch.tensor([[0.8486, 1.1314]], dtype=torch.float16)
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_rejects_invalid_inputs(self) -> None:
        cases = (
            (
                "wrong final dimension",
                4,
                torch.ones(2, 3),
                ValueError,
                "expected final dimension 4",
            ),
            (
                "integer input",
                2,
                torch.ones(1, 2, dtype=torch.int64),
                TypeError,
                "floating-point",
            ),
        )

        for name, hidden_size, inputs, error, message in cases:
            with self.subTest(name), self.assertRaisesRegex(error, message):
                RMSNorm(hidden_size)(inputs)


class Qwen3QKNormTests(unittest.TestCase):
    def test_normalizes_query_and_key_heads_with_independent_weights(self) -> None:
        base = torch.tensor([1.0, 2.0, 3.0, 4.0])
        queries = base.reshape(1, 1, 1, 4).repeat(1, 4, 3, 1)
        keys = base.reshape(1, 1, 1, 4).repeat(1, 2, 3, 1)
        query_norm = RMSNorm(4)
        key_norm = RMSNorm(4)
        with torch.no_grad():
            query_norm.weight.fill_(2.0)
            key_norm.weight.fill_(3.0)

        normalized_queries, normalized_keys = normalize_qwen3_queries_and_keys(
            queries,
            keys,
            query_norm=query_norm,
            key_norm=key_norm,
        )

        self.assertEqual(normalized_queries.shape, (1, 4, 3, 4))
        self.assertEqual(normalized_keys.shape, (1, 2, 3, 4))
        torch.testing.assert_close(
            normalized_keys[0, 0, 0], normalized_queries[0, 0, 0] * 1.5
        )

    def test_rejects_sequence_length_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sequence length"):
            normalize_qwen3_queries_and_keys(
                torch.ones(1, 4, 3, 2),
                torch.ones(1, 2, 4, 2),
                query_norm=RMSNorm(2),
                key_norm=RMSNorm(2),
            )


if __name__ == "__main__":
    unittest.main()
