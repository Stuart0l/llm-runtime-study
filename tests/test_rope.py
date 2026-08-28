from __future__ import annotations

import math
import unittest

import torch

from mini_llm.nn import (
    RotaryEmbedding,
    apply_rotary_position_embeddings,
    build_position_ids,
)


class RotaryEmbeddingTests(unittest.TestCase):
    def test_inverse_frequencies_follow_theta_schedule(self) -> None:
        rope = RotaryEmbedding(4, theta=10_000.0)

        torch.testing.assert_close(
            rope.inverse_frequencies,
            torch.tensor([1.0, 0.01]),
        )

    def test_position_zero_is_identity_rotation(self) -> None:
        rope = RotaryEmbedding(4)
        queries = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        keys = queries.clone()
        cosine, sine = rope(torch.tensor([[0]]))

        rotated_queries, rotated_keys = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        torch.testing.assert_close(rotated_queries, queries)
        torch.testing.assert_close(rotated_keys, keys)

    def test_position_one_matches_explicit_half_split_rotation(self) -> None:
        rope = RotaryEmbedding(4, theta=10_000.0)
        vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
        queries = vector.reshape(1, 1, 1, 4)
        cosine, sine = rope(torch.tensor([[1]]))
        expected = torch.tensor(
            [
                vector[0] * math.cos(1.0) - vector[2] * math.sin(1.0),
                vector[1] * math.cos(0.01) - vector[3] * math.sin(0.01),
                vector[2] * math.cos(1.0) + vector[0] * math.sin(1.0),
                vector[3] * math.cos(0.01) + vector[1] * math.sin(0.01),
            ]
        )

        rotated, _ = apply_rotary_position_embeddings(
            queries, queries, cosine, sine
        )

        torch.testing.assert_close(rotated[0, 0, 0], expected)

    def test_rotation_preserves_vector_magnitude(self) -> None:
        rope = RotaryEmbedding(8)
        queries = torch.randn(2, 4, 3, 8)
        keys = torch.randn(2, 2, 3, 8)
        positions = build_position_ids(3, batch_size=2)
        cosine, sine = rope(positions)

        rotated_queries, rotated_keys = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        torch.testing.assert_close(
            rotated_queries.norm(dim=-1), queries.norm(dim=-1)
        )
        torch.testing.assert_close(rotated_keys.norm(dim=-1), keys.norm(dim=-1))

    def test_supports_different_query_and_key_head_counts(self) -> None:
        rope = RotaryEmbedding(8)
        queries = torch.ones(1, 16, 2, 8)
        keys = torch.ones(1, 8, 2, 8)
        cosine, sine = rope(build_position_ids(2))

        rotated_queries, rotated_keys = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        self.assertEqual(rotated_queries.shape, (1, 16, 2, 8))
        self.assertEqual(rotated_keys.shape, (1, 8, 2, 8))

    def test_returns_requested_reduced_precision_dtype(self) -> None:
        rope = RotaryEmbedding(4)

        cosine, sine = rope(torch.tensor([[0, 1]]), output_dtype=torch.bfloat16)

        self.assertEqual(cosine.dtype, torch.bfloat16)
        self.assertEqual(sine.dtype, torch.bfloat16)

    def test_nonzero_offset_matches_explicit_absolute_positions(self) -> None:
        rope = RotaryEmbedding(4)
        offset_positions = build_position_ids(3, offset=10)

        offset_cosine, offset_sine = rope(offset_positions)
        explicit_cosine, explicit_sine = rope(torch.tensor([[10, 11, 12]]))

        torch.testing.assert_close(offset_cosine, explicit_cosine)
        torch.testing.assert_close(offset_sine, explicit_sine)

    def test_rejects_positions_outside_model_limit(self) -> None:
        rope = RotaryEmbedding(4, max_position_embeddings=8)

        with self.assertRaisesRegex(ValueError, "exceeds the model limit"):
            rope(torch.tensor([[8]]))

    def test_rejects_negative_position(self) -> None:
        rope = RotaryEmbedding(4)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            rope(torch.tensor([[-1]]))


class PositionIdTests(unittest.TestCase):
    def test_builds_batched_absolute_positions(self) -> None:
        positions = build_position_ids(3, offset=5, batch_size=2)

        torch.testing.assert_close(
            positions,
            torch.tensor([[5, 6, 7], [5, 6, 7]]),
        )


if __name__ == "__main__":
    unittest.main()
