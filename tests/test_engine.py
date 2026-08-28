from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from mini_llm.engine import Engine, EngineError, resolve_device, resolve_dtype
from mini_llm.sampling import SamplingConfig


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"


class DeviceAndDtypeTests(unittest.TestCase):
    def test_auto_prefers_mps_when_available(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), torch.device("mps"))

    def test_auto_falls_back_to_cpu(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_explicit_unavailable_mps_has_clear_error(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(EngineError, "MPS.*unavailable"),
        ):
            resolve_device("mps")

    def test_rejects_device_outside_v1_scope(self) -> None:
        with self.assertRaisesRegex(EngineError, "supports cpu and mps"):
            resolve_device("cuda")

    def test_auto_dtype_is_fp16_on_mps_and_fp32_on_cpu(self) -> None:
        self.assertEqual(
            resolve_dtype("auto", device=torch.device("mps")), torch.float16
        )
        self.assertEqual(
            resolve_dtype("auto", device=torch.device("cpu")), torch.float32
        )

    def test_accepts_explicit_dtype_string_or_torch_dtype(self) -> None:
        self.assertEqual(
            resolve_dtype("bfloat16", device=torch.device("cpu")), torch.bfloat16
        )
        self.assertEqual(
            resolve_dtype(torch.float16, device=torch.device("cpu")), torch.float16
        )


class EngineTests(unittest.TestCase):
    @patch("mini_llm.engine.synchronize_device")
    @patch("mini_llm.engine.Qwen3Tokenizer.from_model_dir")
    @patch("mini_llm.engine.Qwen3ForCausalLM.from_model_dir")
    def test_from_model_dir_places_and_freezes_model_once(
        self,
        load_model: MagicMock,
        load_tokenizer: MagicMock,
        synchronize: MagicMock,
    ) -> None:
        model = MagicMock()
        model.to.return_value = model
        load_model.return_value = model
        tokenizer = MagicMock()
        load_tokenizer.return_value = tokenizer

        engine = Engine.from_model_dir(
            Path("model"), device="cpu", dtype="float32", max_seq_len=128
        )

        model.config.validate_context_length.assert_called_once_with(128)
        model.requires_grad_.assert_called_once_with(False)
        model.eval.assert_called_once_with()
        model.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.float32)
        model.model.rotary_emb.materialize.assert_called_once_with(torch.device("cpu"))
        synchronize.assert_called_once_with(torch.device("cpu"))
        self.assertIs(engine.model, model)
        self.assertIs(engine.tokenizer, tokenizer)

    @patch("mini_llm.engine.generate_text")
    def test_generate_forwards_engine_context_and_sampling(
        self, generate_text: MagicMock
    ) -> None:
        generate_text.return_value = iter(())
        model = MagicMock()
        tokenizer = MagicMock()
        engine = Engine(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_seq_len=256,
            load_seconds=1.0,
        )
        sampling = SamplingConfig(temperature=0.7, top_k=20, seed=4)

        result = engine.generate(
            "raw prompt",
            max_new_tokens=12,
            sampling=sampling,
            enable_thinking=True,
        )

        self.assertIs(result, generate_text.return_value)
        generate_text.assert_called_once_with(
            model,
            tokenizer,
            "raw prompt",
            max_new_tokens=12,
            sampling=sampling,
            enable_thinking=True,
            max_seq_len=256,
            synchronize=engine.synchronize,
        )

@unittest.skipUnless(
    torch.backends.mps.is_available() and MODEL_DIR.is_dir(),
    "MPS or local Qwen3 checkpoint is unavailable",
)
class MPSEngineIntegrationTests(unittest.TestCase):
    def test_fp16_model_inputs_rope_and_cache_run_on_mps(self) -> None:
        engine = Engine.from_model_dir(
            MODEL_DIR, device="mps", dtype="auto", max_seq_len=128
        )

        events = list(engine.generate("Say hello.", max_new_tokens=2))

        self.assertEqual(engine.device, torch.device("mps"))
        self.assertEqual(engine.dtype, torch.float16)
        self.assertTrue(events)
        self.assertEqual(engine.model.model.embed_tokens.weight.device.type, "mps")
        self.assertEqual(engine.model.model.rotary_emb.inverse_frequencies.device.type, "mps")
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.dtype, torch.float32
        )
        self.assertIsNotNone(engine.model.cache)
        self.assertEqual(engine.model.cache.device.type, "mps")


if __name__ == "__main__":
    unittest.main()
