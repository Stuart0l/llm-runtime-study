from __future__ import annotations

import unittest

import torch

from mini_llm.cache import DenseKVCache, KVCacheError, LayerKVCache
from mini_llm.config import GraniteMoeConfig
from mini_llm.model import Qwen3ForCausalLM
from tests.test_config import valid_granite_config
from tests.test_model import _tiny_config


class LayerKVCacheTests(unittest.TestCase):
    def test_append_returns_only_valid_prefix_and_reset_reuses_storage(self) -> None:
        cache = LayerKVCache(
            keys=torch.empty(1, 2, 4, 3),
            values=torch.empty(1, 2, 4, 3),
        )
        keys = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
        values = keys + 100
        key_pointer = cache.keys.data_ptr()

        cached_keys, cached_values = cache.append(keys, values)

        self.assertEqual(cache.length, 2)
        self.assertEqual(cached_keys.shape, (1, 2, 2, 3))
        torch.testing.assert_close(cached_keys, keys)
        torch.testing.assert_close(cached_values, values)

        cache.reset()

        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.keys.data_ptr(), key_pointer)

    def test_rejects_context_overflow_before_writing(self) -> None:
        cache = LayerKVCache(
            keys=torch.empty(1, 1, 2, 2),
            values=torch.empty(1, 1, 2, 2),
        )
        cache.append(torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))

        with self.assertRaisesRegex(KVCacheError, "capacity exceeded"):
            cache.append(torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))

        self.assertEqual(cache.length, 2)


class DenseKVCacheTests(unittest.TestCase):
    def test_allocates_from_granite_decoder_config(self) -> None:
        config = GraniteMoeConfig.from_dict(valid_granite_config())

        cache = DenseKVCache(
            config, capacity=2, dtype=torch.float16, device="cpu"
        )

        self.assertEqual(len(cache.layers), 24)
        self.assertEqual(cache.layers[0].keys.shape, (1, 8, 2, 64))
        self.assertEqual(cache.num_bytes, config.kv_cache_bytes(2))

    def test_allocates_one_original_kv_head_pair_per_layer(self) -> None:
        config = _tiny_config()
        cache = DenseKVCache(
            config, capacity=6, dtype=torch.float32, device="cpu"
        )

        self.assertEqual(len(cache.layers), config.num_hidden_layers)
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 6, 2))
        expected_bytes = config.kv_cache_bytes(6, dtype="float32")
        self.assertEqual(cache.num_bytes, expected_bytes)

    def test_prefill_and_single_token_decode_match_uncached_logits(self) -> None:
        torch.manual_seed(23)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        all_tokens = torch.tensor([[1, 4, 7, 9, 11]])
        model.setup_cache(capacity=5)

        with torch.inference_mode():
            reference = model(all_tokens)
            prefill = model.prefill(all_tokens[:, :3])
            decode_1 = model.decode(all_tokens[:, 3:4])
            decode_2 = model.decode(all_tokens[:, 4:5])

        torch.testing.assert_close(prefill, reference[:, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            decode_1, reference[:, 3:4], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            decode_2, reference[:, 4:5], rtol=1e-5, atol=1e-6
        )
        self.assertIsNotNone(model.cache)
        self.assertEqual(model.cache.length, 5)
        self.assertTrue(all(layer.length == 5 for layer in model.cache.layers))

    def test_chunked_cached_attention_uses_absolute_causal_mask(self) -> None:
        torch.manual_seed(29)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        tokens = torch.tensor([[1, 4, 7, 9]])
        model.setup_cache(capacity=4)

        with torch.inference_mode():
            reference = model(tokens)
            first_chunk = model.prefill(tokens[:, :2])
            # Qwen3Model's internal cache path also supports chunked appends,
            # even though the public decode API intentionally accepts one token.
            second_chunk = model.model(
                tokens[:, 2:],
                layer_caches=model.cache.layers,
                position_offset=model.cache.length,
            )
            second_chunk = model.lm_head(second_chunk)

        torch.testing.assert_close(
            first_chunk, reference[:, :2], rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            second_chunk, reference[:, 2:], rtol=1e-5, atol=1e-6
        )

    def test_model_checks_overflow_before_any_layer_is_modified(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=3)

        with torch.inference_mode():
            model.prefill(torch.tensor([[1, 2, 3]]))
            with self.assertRaisesRegex(KVCacheError, "capacity exceeded"):
                model.decode(torch.tensor([[4]]))

        self.assertIsNotNone(model.cache)
        self.assertEqual(model.cache.length, 3)
        self.assertTrue(all(layer.length == 3 for layer in model.cache.layers))

    def test_cached_positions_must_continue_from_logical_length(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=4)

        with torch.inference_mode():
            model.prefill(torch.tensor([[1, 2]]))
            with self.assertRaisesRegex(ValueError, "continue from the cache length 2"):
                model.model(
                    torch.tensor([[3]]),
                    position_ids=torch.tensor([[0]]),
                    layer_caches=model.cache.layers,
                    position_offset=model.cache.length,
                )

        self.assertIsNotNone(model.cache)
        self.assertEqual(model.cache.length, 2)

    def test_prefill_starts_a_new_prompt_without_reallocation(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=4)
        self.assertIsNotNone(model.cache)
        pointer = model.cache.layers[0].keys.data_ptr()

        with torch.inference_mode():
            first = model.prefill(torch.tensor([[1, 2]]))
            second = model.prefill(torch.tensor([[1, 2]]))

        self.assertEqual(model.cache.length, 2)
        self.assertEqual(model.cache.layers[0].keys.data_ptr(), pointer)
        torch.testing.assert_close(first, second)

    def test_setup_uses_decoder_dtype_and_device_once(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=4)

        self.assertEqual(model.cache.dtype, model.model.embed_tokens.weight.dtype)
        self.assertEqual(model.cache.device, model.model.embed_tokens.weight.device)
        self.assertFalse(hasattr(model.model, "cache"))
        self.assertFalse(hasattr(model.model, "setup_cache"))

    def test_setup_reuses_a_large_enough_cache_for_the_next_request(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=4)

        with torch.inference_mode():
            model.prefill(torch.tensor([[1, 2]]))
        cache = model.cache
        assert cache is not None
        key_pointer = cache.layers[0].keys.data_ptr()

        model.setup_cache(capacity=3)

        self.assertIs(model.cache, cache)
        self.assertEqual(model.cache.capacity, 4)
        self.assertEqual(model.cache.length, 0)
        self.assertEqual(model.cache.layers[0].keys.data_ptr(), key_pointer)

    def test_setup_replaces_an_undersized_cache(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=3)
        cache = model.cache
        assert cache is not None

        model.setup_cache(capacity=5)

        self.assertIsNot(model.cache, cache)
        self.assertEqual(model.cache.capacity, 5)
        self.assertEqual(model.cache.length, 0)

    def test_device_or_dtype_change_releases_owned_cache(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        model.setup_cache(capacity=4)

        model.to(dtype=torch.float64)

        self.assertIsNone(model.cache)

    def test_public_cached_execution_requires_setup_then_prefill(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()

        with self.assertRaisesRegex(RuntimeError, "setup_cache.*before decode"):
            model.decode(torch.tensor([[1]]))

        model.setup_cache(capacity=4)
        self.assertIsNotNone(model.cache)
        self.assertEqual(model.cache.length, 0)

        with self.assertRaisesRegex(RuntimeError, "prefill.*before decode"):
            model.decode(torch.tensor([[1]]))

        with torch.inference_mode():
            model.prefill(torch.tensor([[1, 2]]))
            model(torch.tensor([[3, 4]]))

        self.assertEqual(model.cache.length, 2)


if __name__ == "__main__":
    unittest.main()
