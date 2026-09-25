from __future__ import annotations

import unittest

import torch

from mini_llm.cache import CacheAllocation, KVCacheError
from mini_llm.cache.dense import DenseKVCacheManager
from mini_llm.config import Qwen3Config
from mini_llm.cache.paged import PagedBatchLayerKVCache, PagedKVCacheManager
from mini_llm.model.qwen import Qwen3ForCausalLM
from tests.test_qwen_model import _tiny_config, _tiny_config_data


class DenseSequenceKVCacheTests(unittest.TestCase):
    def test_initializes_unwritten_storage_to_zero(self) -> None:
        manager = DenseKVCacheManager(
            _tiny_config(), 3, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(3)

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


class CacheBackendContractTests(unittest.TestCase):
    def _managers(self, capacity: int):
        config = _tiny_config()
        return (
            DenseKVCacheManager(
                config, capacity, dtype=torch.float32, device="cpu"
            ),
            PagedKVCacheManager(
                config, capacity, dtype=torch.float32, device="cpu"
            ),
        )

    def test_last_allocation_reports_reserved_bytes_for_each_backend(self) -> None:
        config = _tiny_config()
        for cls, reserved_tokens in (
            (DenseKVCacheManager, 5),
            (PagedKVCacheManager, 16),
        ):
            with self.subTest(manager=cls.__name__):
                manager = cls(config, 32, dtype=torch.float16, device="cpu")
                manager.allocate(5)
                self.assertEqual(
                    manager.last_allocation,
                    CacheAllocation(
                        num_bytes=config.kv_cache_bytes(
                            reserved_tokens, dtype="float16"
                        ),
                        capacity=5,
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
                    decode_1 = model.decode(tokens[:, 3:4], caches=(cache,))
                    decode_2 = model.decode(tokens[:, 4:5], caches=(cache,))
                    torch.testing.assert_close(prefill, reference[:, 2:3])
                    torch.testing.assert_close(decode_1, reference[:, 3:4])
                    torch.testing.assert_close(decode_2, reference[:, 4:5])
                    self.assertEqual(cache.length, 5)
                    manager.release(cache)
                    self.assertEqual(manager.active_sequences, 0)

    def test_ragged_decode_batch_matches_independent_sequences(self) -> None:
        torch.manual_seed(43)
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        first_tokens = torch.tensor([[1, 4, 7]])
        second_tokens = torch.tensor([[2, 5, 8, 10, 12]])

        with torch.inference_mode():
            first_reference = model(first_tokens)
            second_reference = model(second_tokens)
            for manager in self._managers(32):
                with self.subTest(manager=type(manager).__name__):
                    first_cache = manager.allocate(6)
                    second_cache = manager.allocate(6)
                    model.prefill(first_tokens[:, :2], cache=first_cache)
                    model.prefill(second_tokens[:, :4], cache=second_cache)

                    logits = model.decode(
                        torch.tensor([[7], [12]]),
                        caches=(first_cache, second_cache),
                    )

                    torch.testing.assert_close(logits[0], first_reference[:, 2])
                    torch.testing.assert_close(logits[1], second_reference[:, 4])
                    self.assertEqual(first_cache.length, 3)
                    self.assertEqual(second_cache.length, 5)
                    manager.release(first_cache)
                    manager.release(second_cache)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_ragged_paged_decode_matches_uncached_sequences(self) -> None:
        config_data = _tiny_config_data()
        config_data.update(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
        )
        model = (
            Qwen3ForCausalLM(Qwen3Config.from_dict(config_data))
            .cuda()
            .half()
            .eval()
        )
        model.requires_grad_(False)
        model.materialize_derived_buffers(torch.device("cuda"))
        first_tokens = torch.tensor([[1, 4, 7]], device="cuda")
        second_tokens = torch.tensor([[2, 5, 8, 10, 12]], device="cuda")
        pool = PagedKVCacheManager(
            model.config, 32, dtype=torch.float16, device="cuda"
        )
        first = pool.allocate(6)
        second = pool.allocate(6)

        with torch.inference_mode():
            first_reference = model(first_tokens)
            second_reference = model(second_tokens)
            model.prefill(first_tokens[:, :2], cache=first)
            model.prefill(second_tokens[:, :4], cache=second)
            logits = model.decode(
                torch.tensor([[7], [12]], device="cuda"),
                caches=(first, second),
            )

        torch.testing.assert_close(
            logits[0], first_reference[:, 2], rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            logits[1], second_reference[:, 4], rtol=1e-2, atol=1e-2
        )
        pool.release(first)
        pool.release(second)

    def test_overflow_is_checked_before_any_layer_is_modified(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        for manager in self._managers(3):
            with self.subTest(manager=type(manager).__name__):
                cache = manager.allocate(3)
                with torch.inference_mode():
                    model.prefill(torch.tensor([[1, 2, 3]]), cache=cache)
                    with self.assertRaisesRegex(KVCacheError, "capacity exceeded"):
                        model.decode(torch.tensor([[4]]), caches=(cache,))
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
            model.decode(torch.tensor([[1]]), caches=(cache,))

    def test_prefill_rejects_batches_larger_than_one(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = DenseKVCacheManager(
            model.config, 4, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(4)

        with self.assertRaisesRegex(ValueError, "batch size one"):
            model.prefill(torch.tensor([[1], [2]]), cache=cache)

    def test_model_rejects_backend_neutrally_incompatible_cache(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = DenseKVCacheManager(
            model.config, 4, dtype=torch.float64, device="cpu"
        )
        cache = manager.allocate(4)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            model.prefill(torch.tensor([[1, 2]]), cache=cache)


class PagedKVCacheManagerTests(unittest.TestCase):
    def test_write_and_gather_cross_block_boundaries(self) -> None:
        config = _tiny_config()
        manager = PagedKVCacheManager(
            config, 6, block_size=2, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(5)
        self.assertEqual(cache.block_table.tolist(), [0, 1, 2])
        self.assertEqual(cache.block_table.dtype, torch.int32)
        self.assertEqual(cache.block_table.device, manager.spec.device)
        block_table_pointer = cache.block_table.data_ptr()
        first = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
        second = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2) + 20
        expected = torch.cat((first, second), dim=2)
        cache.extend(3)
        for layer in cache.layers:
            layer.write(first, first + 100, torch.tensor([[0, 1, 2]]))
        cache.extend(2)
        for layer in cache.layers:
            layer.write(second, second + 100, torch.tensor([[3, 4]]))

        gathered = cache.layers[0].gathered()
        paged = cache.layers[0].paged()
        torch.testing.assert_close(gathered.keys, expected)
        torch.testing.assert_close(gathered.values, expected + 100)
        self.assertIs(paged.keys, manager.store.keys[0])
        self.assertIs(paged.values, manager.store.values[0])
        self.assertEqual(paged.max_seqlen_k, 6)
        self.assertEqual(cache.block_table.data_ptr(), block_table_pointer)
        manager.release(cache)

    def test_batched_write_matches_single_request_write(self) -> None:
        manager = PagedKVCacheManager(
            _tiny_config(), 16, block_size=2, dtype=torch.float32, device="cpu"
        )
        for tensor in (*manager.store.keys, *manager.store.values):
            tensor.zero_()
        first = manager.allocate(4)
        second = manager.allocate(4)
        keys = torch.arange(12, dtype=torch.float32).view(1, 2, 3, 2)
        positions = torch.tensor([[0, 1, 2]])
        for cache in (first, second):
            cache.extend(3)
            cache.layers[0].write(keys, keys + 100, positions)
        single_keys = manager.store.keys[0].clone()
        single_values = manager.store.values[0].clone()

        manager.store.keys[0].zero_()
        manager.store.values[0].zero_()
        batch = PagedBatchLayerKVCache(manager.store, 0, (first, second))
        batched = torch.cat((keys, keys))
        batch.write(batched, batched + 100, torch.tensor([[0, 1, 2], [0, 1, 2]]))

        torch.testing.assert_close(manager.store.keys[0], single_keys)
        torch.testing.assert_close(manager.store.values[0], single_values)

    def test_pool_exhaustion_release_and_deterministic_reuse(self) -> None:
        manager = PagedKVCacheManager(
            _tiny_config(), 4, block_size=2, dtype=torch.float32, device="cpu"
        )
        first = manager.allocate(2)
        second = manager.allocate(2)
        with self.assertRaisesRegex(KVCacheError, "pool exhausted"):
            manager.allocate(1)
        released_id = first.block_table[0].item()
        manager.release(first)
        manager.release(first)
        replacement = manager.allocate(1)
        self.assertEqual(replacement.block_table.tolist(), [released_id])
        self.assertEqual(second.block_table.tolist(), [1])

    def test_reset_rollback_release_and_block_reuse(self) -> None:
        config = _tiny_config()
        manager = PagedKVCacheManager(
            config, 4, block_size=2, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(4)
        cache.extend(1)
        self.assertEqual(cache.length, 1)
        self.assertTrue(all(layer.length == 1 for layer in cache.layers))
        cache.reset()
        states = torch.ones(1, 2, 2, 2)
        cache.extend(2)
        for layer in cache.layers:
            layer.write(states, states, torch.tensor([[0, 1]]))
        blocks = cache.block_table.clone()
        cache.rollback(1)
        cache.reset()
        torch.testing.assert_close(cache.block_table, blocks)
        manager.release(cache)
        with self.assertRaisesRegex(KVCacheError, "released"):
            _ = cache.length
        self.assertEqual(manager.used_blocks, 0)

    def test_model_failure_rolls_back_partially_written_layers(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config()).eval()
        manager = PagedKVCacheManager(
            model.config, 4, dtype=torch.float32, device="cpu"
        )
        cache = manager.allocate(4)
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


if __name__ == "__main__":
    unittest.main()