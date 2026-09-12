from __future__ import annotations

import gc
from pathlib import Path
import unittest

import torch

from mini_llm.cache.paged import PagedKVCachePool
from mini_llm.config import Qwen3Config
from mini_llm.cuda_graph import PagedDecodeGraph
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
        pool = PagedKVCachePool(config, 96, dtype=torch.float16, device="cuda")
        replay_cache = pool.allocate(4)
        eager_cache = pool.allocate(4)
        prompt = torch.tensor([[1, 4]], device="cuda")

        with torch.inference_mode():
            model.prefill(prompt, cache=replay_cache)
            model.prefill(prompt, cache=eager_cache)
            graph = PagedDecodeGraph(model, replay_cache)

            for token in (7, 8):
                token_input = torch.tensor([[token]], device="cuda")
                actual = graph.replay(token_input).clone()
                expected = model.decode(token_input, cache=eager_cache)
                torch.testing.assert_close(actual, expected)

        self.assertEqual(replay_cache.length, 4)
        self.assertEqual(eager_cache.length, 4)

        original_first_block = graph.cache.block_table[0].clone()
        pool.release(replay_cache)
        pool.release(eager_cache)
        reserved_cache = pool.allocate(4)
        replay_cache = pool.allocate(20)
        eager_cache = pool.allocate(20)
        prompt = torch.arange(18, device="cuda").remainder(config.vocab_size)[
            None, :
        ]
        self.assertEqual(graph.cache.block_table.numel(), 2)
        self.assertEqual(replay_cache.block_table.numel(), 2)
        self.assertNotEqual(replay_cache.block_table[0], original_first_block)

        with torch.inference_mode():
            model.prefill(prompt, cache=replay_cache)
            model.prefill(prompt, cache=eager_cache)
            graph.cache.bind(replay_cache)
            for token in (6, 9):
                token_input = torch.tensor([[token]], device="cuda")
                actual = graph.replay(token_input).clone()
                expected = model.decode(token_input, cache=eager_cache)
                torch.testing.assert_close(actual, expected)

        pool.release(reserved_cache)
        pool.release(replay_cache)
        pool.release(eager_cache)

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
                pool = PagedKVCachePool(
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
                        graph = PagedDecodeGraph(model, replay_cache)
                        actual = graph.replay(token).clone()
                        expected = model.decode(token, cache=eager_cache)
                    torch.testing.assert_close(actual, expected)
                finally:
                    del graph, replay_cache, eager_cache, pool
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
