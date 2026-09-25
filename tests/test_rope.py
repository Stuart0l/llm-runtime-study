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
    def test_matches_explicit_half_split_rotation(self) -> None:
        rope = RotaryEmbedding(4, theta=10_000.0)
        query = torch.tensor([1.0, 2.0, 3.0, 4.0])
        key = torch.tensor([-0.5, 1.5, 2.5, -3.0])
        queries = query.expand(1, 1, 2, 4).clone()
        keys = key.expand(1, 1, 2, 4).clone()
        cosine, sine = rope(torch.tensor([[0, 1]]))

        def rotate(vector: torch.Tensor, position: int) -> torch.Tensor:
            first, second = position * 1.0, position * 0.01
            return torch.tensor(
                [
                    vector[0] * math.cos(first) - vector[2] * math.sin(first),
                    vector[1] * math.cos(second) - vector[3] * math.sin(second),
                    vector[2] * math.cos(first) + vector[0] * math.sin(first),
                    vector[3] * math.cos(second) + vector[1] * math.sin(second),
                ]
            )

        rotated_queries, rotated_keys = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )

        torch.testing.assert_close(rotated_queries[0, 0, 0], query)
        torch.testing.assert_close(rotated_keys[0, 0, 0], key)
        torch.testing.assert_close(rotated_queries[0, 0, 1], rotate(query, 1))
        torch.testing.assert_close(rotated_keys[0, 0, 1], rotate(key, 1))

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

    def test_returns_requested_reduced_precision_dtype(self) -> None:
        rope = RotaryEmbedding(4)

        cosine, sine = rope(torch.tensor([[0, 1]]), output_dtype=torch.bfloat16)

        self.assertEqual(cosine.dtype, torch.bfloat16)
        self.assertEqual(sine.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_forward_is_cuda_graph_capturable(self) -> None:
        rope = RotaryEmbedding(128).cuda()
        position_ids = torch.tensor([[17]], device="cuda")

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                rope(position_ids, output_dtype=torch.float16)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = rope(position_ids, output_dtype=torch.float16)
        graph.replay()
        expected = rope(position_ids, output_dtype=torch.float16)

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


class PositionIdTests(unittest.TestCase):
    def test_builds_batched_absolute_positions(self) -> None:
        positions = build_position_ids(3, offset=5, batch_size=2)

        torch.testing.assert_close(
            positions,
            torch.tensor([[5, 6, 7], [5, 6, 7]]),
        )


if __name__ == "__main__":
    unittest.main()
