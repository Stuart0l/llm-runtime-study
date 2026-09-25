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


def _manual_gqa_reference(
    attention: Qwen3Attention | GraniteAttention,
    inputs: torch.Tensor,
    *,
    normalize_qk: bool,
    scale: float,
) -> torch.Tensor:
    batch_size, sequence_length, _ = inputs.shape
    heads = attention.num_attention_heads
    kv_heads = attention.num_key_value_heads
    head_dim = attention.head_dim

    queries = F.linear(inputs, attention.q_proj.weight)
    queries = queries.view(batch_size, sequence_length, heads, head_dim).transpose(1, 2)
    keys = F.linear(inputs, attention.k_proj.weight)
    keys = keys.view(batch_size, sequence_length, kv_heads, head_dim).transpose(1, 2)
    values = F.linear(inputs, attention.v_proj.weight)
    values = values.view(batch_size, sequence_length, kv_heads, head_dim).transpose(1, 2)
    if normalize_qk:
        queries = queries * torch.rsqrt(
            queries.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        keys = keys * torch.rsqrt(
            keys.square().mean(dim=-1, keepdim=True) + 1e-6
        )
    keys = keys.repeat_interleave(heads // kv_heads, dim=1)
    values = values.repeat_interleave(heads // kv_heads, dim=1)
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    attended = torch.matmul(probabilities, values)
    attended = attended.transpose(1, 2).contiguous().view(
        batch_size, sequence_length, heads * head_dim
    )
    return F.linear(attended, attention.o_proj.weight)


class RepeatKVHeadsTests(unittest.TestCase):
    def test_repeats_each_head_as_one_contiguous_query_group(self) -> None:
        states = torch.tensor([[[[1.0]], [[2.0]]]])

        repeated = repeat_kv_heads(states, repeats=2)

        torch.testing.assert_close(
            repeated[:, :, 0, 0], torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        )


class GroupedQueryAttentionTests(unittest.TestCase):
    def test_matches_manual_grouped_query_causal_attention(self) -> None:
        cases = (
            (
                Qwen3Attention,
                11,
                {
                    "hidden_size": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "rms_norm_eps": 1e-6,
                },
                True,
                1 / math.sqrt(2),
            ),
            (
                GraniteAttention,
                31,
                {
                    "hidden_size": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                    "attention_scale": 0.25,
                },
                False,
                0.25,
            ),
        )
        for attention_class, seed, kwargs, normalize_qk, scale in cases:
            with self.subTest(attention=attention_class.__name__):
                torch.manual_seed(seed)
                attention = attention_class(**kwargs).eval()
                inputs = torch.tensor([[[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]]])
                cosine, sine = _identity_rope_tables(1, 3, 2)

                expected = _manual_gqa_reference(
                    attention, inputs, normalize_qk=normalize_qk, scale=scale
                )
                actual = attention(inputs, cosine, sine)

                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_invalid_constructor_arguments(self) -> None:
        granite_kwargs = {
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
        }
        cases = (
            (
                Qwen3Attention,
                {
                    "hidden_size": 8,
                    "num_attention_heads": 3,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                },
                "must be divisible",
            ),
            (
                Qwen3Attention,
                {
                    "hidden_size": 8,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 3,
                },
                "head_dim must be even",
            ),
            *(
                (
                    GraniteAttention,
                    {**granite_kwargs, "attention_scale": scale},
                    "attention_scale",
                )
                for scale in (0.0, -1.0, float("inf"))
            ),
        )
        for attention_class, kwargs, regex in cases:
            with self.subTest(attention=attention_class.__name__, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, regex):
                    attention_class(**kwargs)


class GraniteAttentionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
