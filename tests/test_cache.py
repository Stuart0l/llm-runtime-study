from __future__ import annotations

import unittest

import torch

from mini_llm.cache import KVCacheError
from mini_llm.cache.dense import (
    DenseKVCache,
    DenseKVCacheManager,
    DenseLayerKVCache,
)
from mini_llm.config import GraniteMoeConfig
from mini_llm.cache.paged import PagedKVCachePool
from mini_llm.model.qwen import Qwen3ForCausalLM
from tests.test_config import valid_granite_config
from tests.test_qwen_model import _tiny_config


class DenseLayerKVCacheTests(unittest.TestCase):
    def test_append_returns_only_valid_prefix_and_reset_reuses_storage(self) -> None:
        cache = DenseLayerKVCache(
            keys=torch.empty(1, 2, 4, 3),
            values=torch.empty(1, 2, 4, 3),
        )
        keys = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
        values = keys + 100
        key_pointer = cache.keys.data_ptr()

        cached_keys, cached_values = cache.append(
            keys, values, torch.tensor([[0, 1]])
        )

        self.assertEqual(cache.length, 2)
        torch.testing.assert_close(cached_keys, keys)
        torch.testing.assert_close(cached_values, values)
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.keys.data_ptr(), key_pointer)

    def test_rejects_context_overflow_before_writing(self) -> None:
        cache = DenseLayerKVCache(
            keys=torch.empty(1, 1, 2, 2),
            values=torch.empty(1, 1, 2, 2),
        )
        cache.append(
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 2, 2),
            torch.tensor([[0, 1]]),
        )
        with self.assertRaisesRegex(KVCacheError, "capacity exceeded"):
            cache.append(
                torch.ones(1, 1, 1, 2),
                torch.ones(1, 1, 1, 2),
                torch.tensor([[2]]),
            )
        self.assertEqual(cache.length, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_tensor_position_write_replays_at_a_new_position(self) -> None:
        cache = DenseLayerKVCache(
            keys=torch.zeros(1, 1, 20, 2, device="cuda"),
            values=torch.zeros(1, 1, 20, 2, device="cuda"),
        )
        keys = torch.ones(1, 1, 1, 2, device="cuda")
        values = keys + 1
        position_ids = torch.tensor([[0]], device="cuda")

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                cache.append(keys, values, position_ids)
                cache.reset()
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cached_keys, cached_values = cache.append(keys, values, position_ids)
        self.assertEqual(cached_keys.shape[2], 16)
        self.assertEqual(cached_values.shape[2], 16)
        position_ids.fill_(2)
        keys.fill_(3)
        values.fill_(4)
        graph.replay()

        torch.testing.assert_close(cache.keys[:, :, 2], keys[:, :, 0])
        torch.testing.assert_close(cache.values[:, :, 2], values[:, :, 0])


class DenseKVCacheTests(unittest.TestCase):
    def test_allocates_from_granite_decoder_config(self) -> None:
        config = GraniteMoeConfig.from_dict(valid_granite_config())
        cache = DenseKVCache(
            config, capacity=2, dtype=torch.float16, device="cpu"
        )
        self.assertEqual(len(cache.layers), 24)
        self.assertEqual(cache.layers[0].keys.shape, (1, 8, 2, 64))
        self.assertEqual(cache.num_bytes, config.kv_cache_bytes(2))

    def test_initializes_unwritten_storage_to_zero(self) -> None:
        cache = DenseKVCache(
            _tiny_config(), capacity=3, dtype=torch.float32, device="cpu"
        )

        for layer in cache.layers:
            self.assertEqual(torch.count_nonzero(layer.keys).item(), 0)
            self.assertEqual(torch.count_nonzero(layer.values).item(), 0)

    def test_manager_enforces_aggregate_budget_and_releases(self) -> None:
        manager = DenseKVCacheManager(
            _tiny_config(), 6, dtype=torch.float32, device="cpu"
        )
        first = manager.allocate(4)
        second = manager.allocate(2)
        self.assertEqual(manager.active_sequences, 2)
        self.assertEqual(manager.used_tokens, 6)
        with self.assertRaisesRegex(KVCacheError, "capacity exhausted"):
            manager.allocate(1)
        manager.release(first)
        manager.release(first)
        self.assertEqual(manager.active_sequences, 1)
        replacement = manager.allocate(4)
        self.assertEqual(replacement.capacity, 4)
        manager.release(second)
        manager.release(replacement)
        self.assertEqual(manager.used_tokens, 0)

    def test_advance_updates_all_layer_lengths_after_graph_replay(self) -> None:
        cache = DenseKVCache(
            _tiny_config(), capacity=3, dtype=torch.float32, device="cpu"
        )

        cache.advance(1)

        self.assertEqual(cache.length, 1)
        self.assertTrue(all(layer.length == 1 for layer in cache.layers))


class CacheBackendContractTests(unittest.TestCase):
    def _managers(self, capacity: int):
        config = _tiny_config()
        return (
            DenseKVCacheManager(
                config, capacity, dtype=torch.float32, device="cpu"
            ),
            PagedKVCachePool(
                config, capacity, dtype=torch.float32, device="cpu"
            ),
        )

    def test_prefill_and_decode_match_uncached_for_both_backends(self) -> None:
        torch.manual_seed(23)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        tokens = torch.tensor([[1, 4, 7, 9, 11]])
        with torch.inference_mode():
            reference = model(tokens)
            for manager in self._managers(5):
                with self.subTest(manager=type(manager).__name__):
                    cache = manager.allocate(5)
                    prefill = model.prefill(tokens[:, :3], cache=cache)
                    decode_1 = model.decode(tokens[:, 3:4], cache=cache)
                    decode_2 = model.decode(tokens[:, 4:5], cache=cache)
                    torch.testing.assert_close(prefill, reference[:, :3])
                    torch.testing.assert_close(decode_1, reference[:, 3:4])
                    torch.testing.assert_close(decode_2, reference[:, 4:5])
                    self.assertEqual(cache.length, 5)
                    manager.release(cache)
                    self.assertEqual(manager.active_sequences, 0)

    def test_chunked_append_matches_uncached_for_both_backends(self) -> None:
        torch.manual_seed(29)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        tokens = torch.tensor([[1, 4, 7, 9]])
        with torch.inference_mode():
            reference = model(tokens)
            for manager in self._managers(4):
                with self.subTest(manager=type(manager).__name__):
                    cache = manager.allocate(4)
                    first = model.prefill(tokens[:, :2], cache=cache)
                    second = model._cached_forward(tokens[:, 2:], cache)
                    torch.testing.assert_close(first, reference[:, :2])
                    torch.testing.assert_close(second, reference[:, 2:])
                    manager.release(cache)

    def test_overflow_is_checked_before_any_layer_is_modified(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        for manager in self._managers(3):
            with self.subTest(manager=type(manager).__name__):
                cache = manager.allocate(3)
                with torch.inference_mode():
                    model.prefill(torch.tensor([[1, 2, 3]]), cache=cache)
                    with self.assertRaisesRegex(KVCacheError, "capacity exceeded"):
                        model.decode(torch.tensor([[4]]), cache=cache)
                self.assertEqual(cache.length, 3)
                self.assertTrue(all(layer.length == 3 for layer in cache.layers))
                manager.release(cache)

    def test_decode_requires_prefill(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = DenseKVCacheManager(
            model.config, 4, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(4)
        with self.assertRaisesRegex(RuntimeError, "prefill"):
            model.decode(torch.tensor([[1]]), cache=cache)

    def test_cached_execution_validates_tokens_before_trusted_execution(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = DenseKVCacheManager(
            model.config, 4, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(4)

        with self.assertRaisesRegex(ValueError, "within vocabulary"):
            model.prefill(torch.tensor([[model.config.vocab_size]]), cache=cache)
        with self.assertRaisesRegex(ValueError, "batch size one"):
            model.prefill(torch.tensor([[1], [2]]), cache=cache)

        self.assertEqual(cache.length, 0)

    def test_model_rejects_backend_neutrally_incompatible_cache(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = DenseKVCacheManager(
            model.config, 4, dtype=torch.float64, device="cpu"
        )
        cache = manager.allocate(4)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            model.prefill(torch.tensor([[1, 2]]), cache=cache)

    def test_model_has_no_cache_ownership_or_backend_state(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        for name in ("cache", "cache_pool", "setup_cache", "allocate_cache"):
            self.assertFalse(hasattr(model, name))
        model.to(dtype=torch.float64)
        self.assertEqual(model.model.embed_tokens.weight.dtype, torch.float64)


class PagedKVCachePoolTests(unittest.TestCase):
    def test_append_and_gather_cross_block_boundaries(self) -> None:
        config = _tiny_config()
        pool = PagedKVCachePool(
            config, 6, block_size=2, dtype=torch.float32, device="cpu"
        )
        handle = pool.allocate(5)
        self.assertEqual(handle.block_table.tolist(), [0, 1, 2])
        self.assertEqual(handle.block_table.dtype, torch.int32)
        self.assertEqual(handle.block_table.device, pool.device)
        block_table_pointer = handle.block_table.data_ptr()
        first = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
        second = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2) + 20
        expected = torch.cat((first, second), dim=2)
        gathered_keys = gathered_values = None
        for layer in handle.layers:
            layer.append(first, first + 100, torch.tensor([[0, 1, 2]]))
            gathered_keys, gathered_values = layer.append(
                second, second + 100, torch.tensor([[3, 4]])
            )
        assert gathered_keys is not None and gathered_values is not None
        torch.testing.assert_close(gathered_keys, expected)
        torch.testing.assert_close(gathered_values, expected + 100)
        gathered = handle.layers[0].view(gathered=True)
        scattered = handle.layers[0].view(gathered=False)
        torch.testing.assert_close(gathered.keys, expected)
        self.assertIsNone(gathered.block_table)
        self.assertIs(scattered.keys, pool.keys[0])
        self.assertIs(scattered.values, pool.values[0])
        assert scattered.block_table is not None
        self.assertEqual(scattered.block_table.data_ptr(), block_table_pointer)
        self.assertEqual(scattered.capacity, 5)
        self.assertEqual(handle.block_table.data_ptr(), block_table_pointer)
        pool.release(handle)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gather_is_cuda_graph_capturable(self) -> None:
        pool = PagedKVCachePool(
            _tiny_config(), 4, block_size=2, dtype=torch.float32, device="cuda"
        )
        handle = pool.allocate(4)

        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                pool._gather(0, handle, 2)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            keys, values = pool._gather(0, handle, 2)
        graph.replay()

        self.assertEqual(keys.shape, (1, 2, 2, 2))
        self.assertEqual(values.shape, keys.shape)

    def test_pool_exhaustion_release_and_deterministic_reuse(self) -> None:
        pool = PagedKVCachePool(
            _tiny_config(), 4, block_size=2, dtype=torch.float32, device="cpu"
        )
        first = pool.allocate(2)
        second = pool.allocate(2)
        with self.assertRaisesRegex(KVCacheError, "pool exhausted"):
            pool.allocate(1)
        released_id = first.block_table[0].item()
        pool.release(first)
        pool.release(first)
        replacement = pool.allocate(1)
        self.assertEqual(replacement.block_table.tolist(), [released_id])
        self.assertEqual(second.block_table.tolist(), [1])

    def test_reset_rollback_release_and_block_reuse(self) -> None:
        config = _tiny_config()
        pool = PagedKVCachePool(
            config, 4, block_size=2, dtype=torch.float32, device="cpu"
        )
        handle = pool.allocate(4)
        handle.advance(1)
        self.assertEqual(handle.length, 1)
        self.assertTrue(all(layer.length == 1 for layer in handle.layers))
        handle.reset()
        states = torch.ones(1, 2, 2, 2)
        for layer in handle.layers:
            layer.append(states, states, torch.tensor([[0, 1]]))
        blocks = handle.block_table.clone()
        handle.rollback(1)
        handle.reset()
        torch.testing.assert_close(handle.block_table, blocks)
        pool.release(handle)
        with self.assertRaisesRegex(KVCacheError, "released"):
            _ = handle.length
        self.assertEqual(pool.used_blocks, 0)

    def test_model_failure_rolls_back_partially_written_layers(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        pool = PagedKVCachePool(
            model.config, 4, dtype=torch.float32, device="cpu"
        )
        cache = pool.allocate(4)
        original_forward = model.model.layers[1].forward

        def fail(*args, **kwargs):
            raise RuntimeError("injected layer failure")

        model.model.layers[1].forward = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "injected layer failure"):
                model.prefill(torch.tensor([[1, 2]]), cache=cache)
        finally:
            model.model.layers[1].forward = original_forward
        self.assertEqual(cache.length, 0)
        self.assertTrue(all(layer.length == 0 for layer in cache.layers))

    def test_two_request_handles_keep_independent_model_state(self) -> None:
        torch.manual_seed(41)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        pool = PagedKVCachePool(
            model.config, 32, dtype=torch.float32, device="cpu"
        )
        first = pool.allocate(5)
        second = pool.allocate(5)
        first_tokens = torch.tensor([[1, 2, 3, 4]])
        second_tokens = torch.tensor([[8, 7, 6]])
        with torch.inference_mode():
            first_reference = model(first_tokens)
            second_reference = model(second_tokens)
            first_prefill = model.prefill(first_tokens[:, :3], cache=first)
            second_prefill = model.prefill(second_tokens[:, :2], cache=second)
            first_decode = model.decode(first_tokens[:, 3:], cache=first)
            second_decode = model.decode(second_tokens[:, 2:], cache=second)
        torch.testing.assert_close(first_prefill, first_reference[:, :3])
        torch.testing.assert_close(second_prefill, second_reference[:, :2])
        torch.testing.assert_close(first_decode, first_reference[:, 3:])
        torch.testing.assert_close(second_decode, second_reference[:, 2:])
        self.assertTrue(
            set(first.block_table.tolist()).isdisjoint(second.block_table.tolist())
        )


if __name__ == "__main__":
    unittest.main()
