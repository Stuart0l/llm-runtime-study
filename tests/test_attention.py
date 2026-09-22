from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

from mini_llm.cache import KVCacheSpec
from mini_llm.cache.dense import DenseLayerKVCache
from mini_llm.cache.lengths import SequenceLength
from mini_llm.cache.paged import PagedKVCacheManager
from mini_llm.nn import GraniteAttention, Qwen3Attention, repeat_kv_heads


def _identity_rope_tables(
    batch_size: int, sequence_length: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, head_dim)
    return torch.ones(shape), torch.zeros(shape)


def _dense_layer(
    *,
    capacity: int,
    head_dim: int,
    heads: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[DenseLayerKVCache, SequenceLength]:
    lengths = SequenceLength(capacity)
    shape = (1, heads, capacity, head_dim)
    keys = torch.zeros(shape, device=device, dtype=dtype)
    spec = KVCacheSpec(
        num_layers=1,
        num_key_value_heads=heads,
        head_dim=head_dim,
        dtype=keys.dtype,
        device=keys.device,
    )
    values = torch.zeros(shape, device=device, dtype=dtype)
    return DenseLayerKVCache(keys, values, lengths, spec), lengths


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


class GraniteAttentionTests(unittest.TestCase):
    def test_projection_layout_has_no_q_or_k_norm_weights(self) -> None:
        attention = GraniteAttention(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_scale=0.015625,
        )

        self.assertEqual(attention.q_proj.weight.shape, (8, 8))
        self.assertEqual(attention.k_proj.weight.shape, (4, 8))
        self.assertEqual(attention.v_proj.weight.shape, (4, 8))
        self.assertEqual(attention.o_proj.weight.shape, (8, 8))
        self.assertNotIn("q_norm.weight", attention.state_dict())
        self.assertNotIn("k_norm.weight", attention.state_dict())
        self.assertEqual(attention.scaling, 0.015625)

    def test_matches_manual_gqa_without_query_key_normalization(self) -> None:
        torch.manual_seed(31)
        attention = GraniteAttention(
            hidden_size=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            attention_scale=0.25,
        ).eval()
        inputs = torch.tensor([[[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]])
        cosine, sine = _identity_rope_tables(1, 3, 2)

        queries = F.linear(inputs, attention.q_proj.weight)
        queries = queries.view(1, 3, 2, 2).transpose(1, 2)
        keys = F.linear(inputs, attention.k_proj.weight)
        keys = keys.view(1, 3, 1, 2).transpose(1, 2)
        values = F.linear(inputs, attention.v_proj.weight)
        values = values.view(1, 3, 1, 2).transpose(1, 2)
        keys = keys.repeat_interleave(2, dim=1)
        values = values.repeat_interleave(2, dim=1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * 0.25
        causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, values)
        attended = attended.transpose(1, 2).contiguous().view(1, 3, 4)
        expected = F.linear(attended, attention.o_proj.weight)

        actual = attention(inputs, cosine, sine)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_cached_prefill_and_decode_match_uncached_attention(self) -> None:
        torch.manual_seed(37)
        attention = GraniteAttention(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            attention_scale=0.5,
        ).eval()
        inputs = torch.randn(1, 3, 4)
        cosine, sine = _identity_rope_tables(1, 3, 2)
        cache, lengths = _dense_layer(capacity=3, head_dim=2)

        reference = attention(inputs, cosine, sine)
        lengths.extend(2)
        prefill = attention(
            inputs[:, :2],
            cosine[:, :2],
            sine[:, :2],
            position_ids=torch.tensor([[0, 1]]),
            cache=cache,
        )
        lengths.extend(1)
        decode = attention(
            inputs[:, 2:],
            cosine[:, 2:],
            sine[:, 2:],
            position_ids=torch.tensor([[2]]),
            cache=cache,
        )

        torch.testing.assert_close(prefill, reference[:, :2])
        torch.testing.assert_close(decode, reference[:, 2:])
        self.assertEqual(cache.length, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_dense_cached_attention_graph_replays_at_a_new_length(self) -> None:
        torch.manual_seed(41)
        attention = GraniteAttention(
            hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            attention_scale=0.125,
        ).cuda().half().eval()
        cache, lengths = _dense_layer(
            capacity=4, head_dim=64, device="cuda", dtype=torch.float16
        )
        inputs = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
        cosine, sine = _identity_rope_tables(1, 1, 64)
        cosine = cosine.cuda().half()
        sine = sine.cuda().half()
        position_ids = torch.tensor([[0]], device="cuda")

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    lengths.extend(1)
                    attention(
                        inputs,
                        cosine,
                        sine,
                        position_ids=position_ids,
                        cache=cache,
                    )
                    lengths.reset()
            torch.cuda.current_stream().wait_stream(side_stream)

            lengths.extend(1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = attention(
                    inputs,
                    cosine,
                    sine,
                    position_ids=position_ids,
                    cache=cache,
                )
            position_ids.fill_(2)
            inputs.fill_(0.25)
            graph.replay()
            replayed = output.clone()

            reference_cache, reference_lengths = _dense_layer(
                capacity=4, head_dim=64, device="cuda", dtype=torch.float16
            )
            reference_cache.keys.copy_(cache.keys)
            reference_cache.values.copy_(cache.values)
            reference_lengths.extend(2)
            reference_lengths.extend(1)
            expected = attention(
                inputs,
                cosine,
                sine,
                position_ids=position_ids,
                cache=reference_cache,
            )
        torch.testing.assert_close(replayed, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_paged_flash_attention_matches_sdpa_and_replays(self) -> None:
        torch.manual_seed(43)
        attention = GraniteAttention(
            hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            attention_scale=0.125,
        ).cuda().half().eval()
        dense_cache, dense_lengths = _dense_layer(
            capacity=20, head_dim=64, device="cuda", dtype=torch.float16
        )
        manager = PagedKVCacheManager(
            SimpleNamespace(
                max_position_embeddings=32,
                num_hidden_layers=1,
                num_key_value_heads=1,
                head_dim=64,
            ),
            32,
            dtype=torch.float16,
            device="cuda",
        )
        paged_sequence = manager.allocate(20)
        paged_cache = paged_sequence.layers[0]
        inputs = torch.randn(1, 17, 128, device="cuda", dtype=torch.float16)
        cosine, sine = _identity_rope_tables(1, 17, 64)
        cosine = cosine.cuda().half()
        sine = sine.cuda().half()
        positions = torch.arange(17, device="cuda").unsqueeze(0)

        dense_lengths.extend(17)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            dense_prefill = attention(
                inputs,
                cosine,
                sine,
                position_ids=positions,
                cache=dense_cache,
            )
        paged_sequence.extend(17)
        paged_prefill = attention(
            inputs,
            cosine,
            sine,
            position_ids=positions,
            cache=paged_cache,
        )

        decode_input = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
        decode_cosine, decode_sine = _identity_rope_tables(1, 1, 64)
        decode_cosine = decode_cosine.cuda().half()
        decode_sine = decode_sine.cuda().half()
        decode_position = torch.tensor([[17]], device="cuda")
        dense_lengths.extend(1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            dense_decode = attention(
                decode_input,
                decode_cosine,
                decode_sine,
                position_ids=decode_position,
                cache=dense_cache,
            )
        paged_sequence.extend(1)
        paged_decode = attention(
            decode_input,
            decode_cosine,
            decode_sine,
            position_ids=decode_position,
            cache=paged_cache,
        )

        torch.testing.assert_close(
            paged_prefill, dense_prefill, rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(
            paged_decode, dense_decode, rtol=2e-3, atol=2e-3
        )

        graph_input = torch.randn(
            1, 1, 128, device="cuda", dtype=torch.float16
        )
        captured_input = graph_input.clone()
        graph_position = torch.tensor([[18]], device="cuda")
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                paged_sequence.extend(1)
                attention(
                    graph_input,
                    decode_cosine,
                    decode_sine,
                    position_ids=graph_position,
                    cache=paged_cache,
                )
                paged_sequence.rollback(18)
        torch.cuda.current_stream().wait_stream(side_stream)

        paged_sequence.extend(1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = attention(
                graph_input,
                decode_cosine,
                decode_sine,
                position_ids=graph_position,
                cache=paged_cache,
            )
        graph_position.fill_(19)
        graph_input.fill_(0.25)
        graph.replay()
        replayed = graph_output.clone()

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            dense_lengths.extend(1)
            attention(
                captured_input,
                decode_cosine,
                decode_sine,
                position_ids=torch.tensor([[18]], device="cuda"),
                cache=dense_cache,
            )
            dense_lengths.extend(1)
            expected = attention(
                graph_input,
                decode_cosine,
                decode_sine,
                position_ids=graph_position,
                cache=dense_cache,
            )
        torch.testing.assert_close(replayed, expected, rtol=2e-3, atol=2e-3)

    def test_rejects_invalid_explicit_attention_scale(self) -> None:
        for scale in (0.0, -1.0, float("inf")):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "attention_scale"):
                    GraniteAttention(
                        hidden_size=4,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=2,
                        attention_scale=scale,
                    )


if __name__ == "__main__":
    unittest.main()
