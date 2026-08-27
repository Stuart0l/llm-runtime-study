from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from mini_llm.nn import Qwen3Attention, repeat_kv_heads


def _identity_rope_tables(
    batch_size: int, sequence_length: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, head_dim)
    return torch.ones(shape), torch.zeros(shape)


class RepeatKVHeadsTests(unittest.TestCase):
    def test_repeats_each_head_as_one_contiguous_query_group(self) -> None:
        states = torch.tensor([[[[1.0]], [[2.0]]]])

        repeated = repeat_kv_heads(states, repeats=2)

        torch.testing.assert_close(
            repeated[:, :, 0, 0], torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        )

    def test_one_repeat_returns_original_tensor(self) -> None:
        states = torch.randn(1, 2, 3, 4)

        self.assertIs(repeat_kv_heads(states, repeats=1), states)


class Qwen3AttentionTests(unittest.TestCase):
    def test_projection_shapes_match_checkpoint_layout(self) -> None:
        attention = Qwen3Attention(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )

        self.assertEqual(attention.q_proj.weight.shape, (16, 8))
        self.assertEqual(attention.k_proj.weight.shape, (8, 8))
        self.assertEqual(attention.v_proj.weight.shape, (8, 8))
        self.assertEqual(attention.o_proj.weight.shape, (8, 16))
        self.assertEqual(attention.q_norm.weight.shape, (4,))
        self.assertEqual(attention.k_norm.weight.shape, (4,))

    def test_matches_manual_grouped_query_causal_attention(self) -> None:
        torch.manual_seed(11)
        attention = Qwen3Attention(
            hidden_size=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            rms_norm_eps=1e-6,
        ).eval()
        inputs = torch.tensor([[[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]])
        cosine, sine = _identity_rope_tables(1, 3, 2)

        queries = F.linear(inputs, attention.q_proj.weight)
        queries = queries.view(1, 3, 2, 2).transpose(1, 2)
        keys = F.linear(inputs, attention.k_proj.weight)
        keys = keys.view(1, 3, 1, 2).transpose(1, 2)
        values = F.linear(inputs, attention.v_proj.weight)
        values = values.view(1, 3, 1, 2).transpose(1, 2)
        queries = queries * torch.rsqrt(
            queries.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        keys = keys * torch.rsqrt(
            keys.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        keys = keys.repeat_interleave(2, dim=1)
        values = values.repeat_interleave(2, dim=1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(2)
        causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, values)
        attended = attended.transpose(1, 2).contiguous().view(1, 3, 4)
        expected = F.linear(attended, attention.o_proj.weight)

        actual = attention(inputs, cosine, sine)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_future_tokens_cannot_change_earlier_outputs(self) -> None:
        torch.manual_seed(13)
        attention = Qwen3Attention(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
        ).eval()
        first_inputs = torch.randn(1, 4, 4)
        changed_inputs = first_inputs.clone()
        changed_inputs[:, -1] = torch.tensor([100.0, -200.0, 300.0, -400.0])
        cosine, sine = _identity_rope_tables(1, 4, 2)

        first_outputs = attention(first_inputs, cosine, sine)
        changed_outputs = attention(changed_inputs, cosine, sine)

        torch.testing.assert_close(first_outputs[:, :-1], changed_outputs[:, :-1])
        self.assertFalse(torch.allclose(first_outputs[:, -1], changed_outputs[:, -1]))

    def test_returns_residual_stream_width(self) -> None:
        attention = Qwen3Attention(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        ).eval()
        inputs = torch.randn(2, 5, 8)
        cosine, sine = _identity_rope_tables(2, 5, 4)

        outputs = attention(inputs, cosine, sine)

        self.assertEqual(outputs.shape, (2, 5, 8))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_rejects_incompatible_head_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            Qwen3Attention(
                hidden_size=8,
                num_attention_heads=3,
                num_key_value_heads=2,
                head_dim=4,
            )

    def test_rejects_odd_head_dimension_required_by_rope(self) -> None:
        with self.assertRaisesRegex(ValueError, "head_dim must be even"):
            Qwen3Attention(
                hidden_size=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=3,
            )

    def test_rejects_wrong_input_shape(self) -> None:
        attention = Qwen3Attention(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        cosine, sine = _identity_rope_tables(1, 3, 4)

        with self.assertRaisesRegex(ValueError, "expected input shape"):
            attention(torch.ones(1, 3, 7), cosine, sine)


if __name__ == "__main__":
    unittest.main()
