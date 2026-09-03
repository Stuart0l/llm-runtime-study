from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from mini_llm.config import Qwen3Config
from mini_llm.engine import Engine, EngineError
from mini_llm.interfaces import ChatMessage


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"


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
    @patch("mini_llm.engine.synchronize_device")
    @patch("mini_llm.engine.load_model")
    @patch("mini_llm.engine.load_tokenizer")
    @patch("mini_llm.engine.load_config")
    def test_loads_places_and_prepares_quantized_model(
        self,
        load_config: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
        synchronize: MagicMock,
        validate_backend: MagicMock,
    ) -> None:
        config = _quantized_config()
        load_config.return_value = config
        model = MagicMock()
        model.input_device = torch.device("cpu")
        model.parameters.return_value = iter([MagicMock(dtype=torch.float16)])
        load_model.return_value = model

        with patch("torch.cuda.is_available", return_value=True):
            engine = Engine.from_model_dir(
                "quantized-model", device="cuda", dtype="auto", max_seq_len=128
            )

        validate_backend.assert_called()
        model.config.validate_context_length.assert_called_once_with(128)
        model.requires_grad_.assert_called_once_with(False)
        model.to.assert_called_once_with(
            device=torch.device("cuda"), dtype=torch.float16
        )
        model.prepare_quantized.assert_called_once_with(torch.device("cuda"))
        model.materialize_derived_buffers.assert_called_once_with(
            torch.device("cuda")
        )
        synchronize.assert_called_once_with(torch.device("cuda"))
        self.assertEqual(engine.quantization, "gptq-marlin")
        self.assertEqual(engine.device, torch.device("cuda"))
        self.assertEqual(engine.dtype, torch.float16)

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

    @patch("mini_llm.engine.validate_gptq_marlin_device")
    @patch("mini_llm.engine.synchronize_device")
    def test_cuda_move_rebuilds_marlin_layout_without_reloading(
        self, synchronize: MagicMock, validate_backend: MagicMock
    ) -> None:
        model = MagicMock()
        engine = Engine(
            model=model,
            tokenizer=MagicMock(),
            device=torch.device("cuda:0"),
            dtype=torch.float16,
            max_seq_len=128,
            load_seconds=1.0,
            quantization="gptq-marlin",
        )

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            engine.to(device="cuda:1")

        validate_backend.assert_called_once_with(torch.device("cuda:1"))
        model.to.assert_called_once_with(
            device=torch.device("cuda:1"), dtype=torch.float16
        )
        model.prepare_quantized.assert_called_once_with(torch.device("cuda:1"))
        synchronize.assert_called_once_with(torch.device("cuda:1"))

    @patch("mini_llm.engine.validate_gptq_marlin_device")
    def test_existing_quantized_placement_is_a_true_noop(
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

        self.assertIs(engine.to(), engine)

        validate_backend.assert_not_called()
        model.to.assert_not_called()
        model.prepare_quantized.assert_not_called()


    @unittest.skipUnless(
        torch.cuda.is_available() and MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_real_checkpoint_generates_through_engine(self) -> None:
        engine = Engine.from_model_dir(
            MODEL_DIR, device="cuda", dtype="auto", max_seq_len=64
        )
        try:
            events = list(
                engine.generate(
                    [ChatMessage("user", "Say hello.")], max_new_tokens=2
                )
            )

            self.assertEqual(engine.quantization, "gptq-marlin")
            self.assertEqual(engine.device.type, "cuda")
            self.assertEqual(engine.dtype, torch.float16)
            self.assertTrue(events)
            self.assertIsNotNone(engine.model.cache)
            self.assertEqual(engine.model.cache.device.type, "cuda")
        finally:
            del engine
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
