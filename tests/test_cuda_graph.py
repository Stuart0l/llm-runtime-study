from __future__ import annotations

import gc
from pathlib import Path
import unittest

import torch

from mini_llm.cache.paged import PagedKVCacheManager
from mini_llm.config import Qwen3Config
from mini_llm.cuda_graph import PagedDecodeGraph, PagedDecodeGraphCache
from mini_llm.model.qwen import Qwen3ForCausalLM
from tests.test_quantized_qwen_model import _config_data


class PagedDecodeGraphTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_replay_matches_eager_and_rebinds_to_a_new_cache(self) -> None:
        config_data = _config_data()
        del config_data["quantization_config"]
        config = Qwen3Config.from_dict(config_data)
        model = Qwen3ForCausalLM(config).cuda().half().eval()
        model.requires_grad_(False)
        model.materialize_derived_buffers(torch.device("cuda"))
        pool = PagedKVCacheManager(config, 96, dtype=torch.float16, device="cuda")
        replay_cache = pool.allocate(4)
        eager_cache = pool.allocate(4)
        prompt = torch.tensor([[1, 4]], device="cuda")

        with torch.inference_mode():
            model.prefill(prompt, cache=replay_cache)
            model.prefill(prompt, cache=eager_cache)
            graph = PagedDecodeGraph(model, (replay_cache,))

            for token in (7, 8):
                token_input = torch.tensor([[token]], device="cuda")
                actual = graph.replay(token_input).clone()
                expected = model.decode(token_input, caches=(eager_cache,))
                torch.testing.assert_close(actual, expected)

        self.assertEqual(replay_cache.length, 4)
        self.assertEqual(eager_cache.length, 4)

        graph_view = graph.cache.layers[0].paged()
        original_first_block = graph_view.block_table[0, 0].clone()
        pool.release(replay_cache)
        pool.release(eager_cache)
        reserved_cache = pool.allocate(4)
        replay_cache = pool.allocate(20)
        eager_cache = pool.allocate(20)
        prompt = torch.arange(18, device="cuda").remainder(config.vocab_size)[
            None, :
        ]
        self.assertEqual(graph_view.block_table.shape, (1, 2))
        self.assertEqual(replay_cache.block_table.numel(), 2)
        self.assertNotEqual(replay_cache.block_table[0], original_first_block)

        with torch.inference_mode():
            model.prefill(prompt, cache=replay_cache)
            model.prefill(prompt, cache=eager_cache)
            graph.cache.bind((replay_cache,))
            for token in (6, 9):
                token_input = torch.tensor([[token]], device="cuda")
                actual = graph.replay(token_input).clone()
                expected = model.decode(token_input, caches=(eager_cache,))
                torch.testing.assert_close(actual, expected)

        pool.release(reserved_cache)
        pool.release(replay_cache)
        pool.release(eager_cache)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_batched_replay_matches_eager_at_different_lengths(self) -> None:
        config_data = _config_data()
        del config_data["quantization_config"]
        config = Qwen3Config.from_dict(config_data)
        model = Qwen3ForCausalLM(config).cuda().half().eval()
        model.requires_grad_(False)
        model.materialize_derived_buffers(torch.device("cuda"))
        pool = PagedKVCacheManager(config, 96, dtype=torch.float16, device="cuda")
        replay_caches = (pool.allocate(6), pool.allocate(6))
        eager_caches = (pool.allocate(6), pool.allocate(6))
        prompts = (
            torch.tensor([[1, 4]], device="cuda"),
            torch.tensor([[2, 5, 3]], device="cuda"),
        )

        with torch.inference_mode():
            for prompt, replay_cache, eager_cache in zip(
                prompts, replay_caches, eager_caches
            ):
                model.prefill(prompt, cache=replay_cache)
                model.prefill(prompt, cache=eager_cache)
            graph = PagedDecodeGraph(model, replay_caches)

            for tokens in ((7, 8), (9, 6)):
                token_inputs = torch.tensor(tokens, device="cuda").unsqueeze(1)
                actual = graph.replay(token_inputs).clone()
                expected = model.decode(token_inputs, caches=eager_caches)
                torch.testing.assert_close(actual, expected)

        self.assertEqual(
            tuple(cache.length for cache in replay_caches),
            tuple(cache.length for cache in eager_caches),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_graph_cache_keeps_one_graph_per_batch_size_and_matches_eager(
        self,
    ) -> None:
        config_data = _config_data()
        del config_data["quantization_config"]
        config = Qwen3Config.from_dict(config_data)
        model = Qwen3ForCausalLM(config).cuda().half().eval()
        model.requires_grad_(False)
        model.materialize_derived_buffers(torch.device("cuda"))
        pool = PagedKVCacheManager(config, 96, dtype=torch.float16, device="cuda")
        r1, r2, e1, e2 = (pool.allocate(8) for _ in range(4))
        prompts = (
            torch.tensor([[1, 4]], device="cuda"),
            torch.tensor([[2, 5, 3]], device="cuda"),
        )

        with torch.inference_mode():
            for prompt, replay_cache, eager_cache in zip(prompts, (r1, r2), (e1, e2)):
                model.prefill(prompt, cache=replay_cache)
                model.prefill(prompt, cache=eager_cache)
            graphs = PagedDecodeGraphCache(model, pool.store)

            def check(replay_caches, eager_caches, tokens):
                token_inputs = torch.tensor(tokens, device="cuda").unsqueeze(1)
                actual = graphs.bind(replay_caches)(token_inputs).clone()
                expected = model.decode(token_inputs, caches=eager_caches)
                torch.testing.assert_close(actual, expected)

            check((r1, r2), (e1, e2), (7, 8))
            pair_graph = graphs.graphs[2]
            check((r2,), (e2,), (9,))
            check((r1, r2), (e1, e2), (6, 5))

        self.assertEqual(sorted(graphs.graphs), [1, 2])
        self.assertIs(graphs.graphs[2], pair_graph)
        self.assertEqual((r1.length, r2.length), (e1.length, e2.length))

    @unittest.skipUnless(
        torch.cuda.is_available()
        and all(
            (Path(__file__).parents[1] / "models" / name).is_dir()
            for name in ("qwen3-0.6b-int4", "qwen3-0.6b-int8")
        ),
        "requires CUDA and the real GPTQ checkpoints",
    )
    def test_zero_warmup_replay_matches_eager_gptq_decode(self) -> None:
        for name in ("qwen3-0.6b-int4", "qwen3-0.6b-int8"):
            with self.subTest(model=name):
                model_dir = Path(__file__).parents[1] / "models" / name
                model = Qwen3ForCausalLM.from_model_dir(model_dir)
                model.to(device="cuda", dtype=torch.float16)
                model.prepare_quantized("cuda")
                model.materialize_derived_buffers(torch.device("cuda"))
                pool = PagedKVCacheManager(
                    model.config, 32, dtype=torch.float16, device="cuda"
                )
                replay_cache = pool.allocate(4)
                eager_cache = pool.allocate(4)
                prompt = torch.tensor([[1, 4]], device="cuda")
                token = torch.tensor([[7]], device="cuda")
                graph = None

                try:
                    with torch.inference_mode():
                        model.prefill(prompt, cache=replay_cache)
                        model.prefill(prompt, cache=eager_cache)
                        graph = PagedDecodeGraph(model, (replay_cache,))
                        actual = graph.replay(token).clone()
                        expected = model.decode(token, caches=(eager_cache,))
                    torch.testing.assert_close(actual, expected)
                finally:
                    del graph, replay_cache, eager_cache, pool
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
