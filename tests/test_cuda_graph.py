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
    def test_replay_matches_eager_decode_and_advances_cache(self) -> None:
        config_data = _config_data()
        del config_data["quantization_config"]
        config = Qwen3Config.from_dict(config_data)
        model = Qwen3ForCausalLM(config).cuda().half().eval()
        model.requires_grad_(False)
        model.materialize_derived_buffers(torch.device("cuda"))
        pool = PagedKVCachePool(
            config, 32, dtype=torch.float16, device="cuda"
        )
        graph_cache = pool.allocate(4)
        eager_cache = pool.allocate(4)
        prompt = torch.tensor([[1, 4]], device="cuda")

        with torch.inference_mode():
            model.prefill(prompt, cache=graph_cache)
            model.prefill(prompt, cache=eager_cache)
            graph = PagedDecodeGraph(model, graph_cache)

            for token in (7, 8):
                token_input = torch.tensor([[token]], device="cuda")
                actual = graph.replay(token_input).clone()
                expected = model.decode(token_input, cache=eager_cache)
                torch.testing.assert_close(actual, expected)

        self.assertEqual(graph_cache.length, 4)
        self.assertEqual(eager_cache.length, 4)

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
                graph_cache = pool.allocate(4)
                eager_cache = pool.allocate(4)
                prompt = torch.tensor([[1, 4]], device="cuda")
                token = torch.tensor([[7]], device="cuda")
                graph = None

                try:
                    with torch.inference_mode():
                        model.prefill(prompt, cache=graph_cache)
                        model.prefill(prompt, cache=eager_cache)
                        graph = PagedDecodeGraph(model, graph_cache)
                        actual = graph.replay(token).clone()
                        expected = model.decode(token, cache=eager_cache)
                    torch.testing.assert_close(actual, expected)
                finally:
                    del graph, graph_cache, eager_cache, pool
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
