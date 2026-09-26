from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from mini_llm.config import Qwen3Config
from mini_llm.engine import Engine, EngineError
from mini_llm.tokenizer import ChatMessage


MODEL_DIRS = (
    (Path(__file__).parents[1] / "models" / "qwen3-0.6b-int4", 4),
    (Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8", 8),
)


def _quantized_config() -> MagicMock:
    config = MagicMock(spec=Qwen3Config)
    config.quantization_config = MagicMock()
    return config


class QuantizedEngineTests(unittest.TestCase):
    @patch("mini_llm.engine.load_model")
    @patch("mini_llm.engine.load_tokenizer")
    @patch("mini_llm.engine.load_config")
    def test_rejects_cpu_before_loading_runtime(
        self,
        load_config: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
    ) -> None:
        load_config.return_value = _quantized_config()

        with self.assertRaisesRegex(EngineError, "requires CUDA"):
            Engine.from_model_dir(
                "quantized-model", device="cpu", dtype="float16"
            )

        load_tokenizer.assert_not_called()
        load_model.assert_not_called()

    @patch("mini_llm.engine.validate_gptq_marlin_device")
    def test_rejects_dtype_change_before_mutating_resident_model(
        self, validate_backend: MagicMock
    ) -> None:
        model = MagicMock()
        engine = Engine(
            model=model,
            tokenizer=MagicMock(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            max_seq_len=128,
            load_seconds=1.0,
            quantization="gptq-marlin",
        )

        with self.assertRaisesRegex(EngineError, "requires float16"):
            engine.to(dtype="bfloat16")

        validate_backend.assert_not_called()
        model.to.assert_not_called()
        model.prepare_quantized.assert_not_called()

    @unittest.skipUnless(
        torch.cuda.is_available() and all(path.is_dir() for path, _ in MODEL_DIRS),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_real_checkpoints_generate_through_engine(self) -> None:
        for model_dir, bits in MODEL_DIRS:
            with self.subTest(bits=bits):
                engine = Engine.from_model_dir(
                    model_dir, device="cuda", dtype="auto", max_seq_len=64
                )
                try:
                    events = list(
                        engine.generate(
                            ([ChatMessage("user", "Say hello.")],), max_new_tokens=2
                        )
                    )
                    assert engine._decode_graphs is not None
                    first_graph = engine._decode_graphs.graphs[1]
                    repeated_events = list(
                        engine.generate(
                            ([ChatMessage("user", "Say hello.")],), max_new_tokens=2
                        )
                    )

                    self.assertEqual(engine.quantization, "gptq-marlin")
                    assert engine.model.config.quantization_config is not None
                    self.assertEqual(
                        engine.model.config.quantization_config.bits, bits
                    )
                    self.assertEqual(engine.device.type, "cuda")
                    self.assertEqual(engine.dtype, torch.float16)
                    self.assertTrue(events)
                    self.assertIsNotNone(first_graph)
                    self.assertEqual(
                        [event.token_id for _, event in repeated_events],
                        [event.token_id for _, event in events],
                    )
                    self.assertIs(engine._decode_graphs.graphs[1], first_graph)
                    self.assertEqual(engine.cache_manager.active_sequences, 0)
                    self.assertEqual(engine.cache_manager.spec.device.type, "cuda")
                finally:
                    del engine
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
