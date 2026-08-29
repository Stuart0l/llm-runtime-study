from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from mini_llm.config import GraniteMoeConfig, Qwen3Config
from mini_llm.engine import Engine, EngineError, resolve_device, resolve_dtype
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


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
    @patch("mini_llm.engine.load_model")
    @patch("mini_llm.engine.load_tokenizer")
    @patch("mini_llm.engine.load_config")
    def test_from_model_dir_places_and_freezes_loaded_model(
        self,
        load_typed_config: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
        synchronize: MagicMock,
    ) -> None:
        model = MagicMock()
        model.to.return_value = model
        load_model.return_value = model
        tokenizer = MagicMock()
        load_tokenizer.return_value = tokenizer
        config = MagicMock(spec=Qwen3Config)
        load_typed_config.return_value = config

        engine = Engine.from_model_dir(
            Path("model"), device="cpu", dtype="float32", max_seq_len=128
        )

        model.config.validate_context_length.assert_called_once_with(128)
        model.requires_grad_.assert_called_once_with(False)
        model.to.assert_called_once_with(device=torch.device("cpu"), dtype=torch.float32)
        model.materialize_derived_buffers.assert_called_once_with(torch.device("cpu"))
        synchronize.assert_called_once_with(torch.device("cpu"))
        self.assertIs(engine.model, model)
        self.assertIs(engine.tokenizer, tokenizer)
        load_tokenizer.assert_called_once_with(Path("model"), model_config=config)
        load_model.assert_called_once_with(Path("model"), model_config=config)

    @patch("mini_llm.engine.synchronize_device")
    @patch("mini_llm.engine.load_model")
    @patch("mini_llm.engine.load_tokenizer")
    @patch("mini_llm.engine.load_config")
    def test_from_model_dir_loads_registered_granite_model(
        self,
        load_typed_config: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
        _synchronize: MagicMock,
    ) -> None:
        config = MagicMock(spec=GraniteMoeConfig)
        load_typed_config.return_value = config
        model = MagicMock()
        model.to.return_value = model
        load_model.return_value = model

        engine = Engine.from_model_dir(
            "granite", device="cpu", dtype="bfloat16", max_seq_len=128
        )

        self.assertIs(engine.model, model)
        load_tokenizer.assert_called_once_with("granite", model_config=config)
        load_model.assert_called_once_with("granite", model_config=config)

    @patch("mini_llm.engine.generate_text")
    def test_generate_forwards_complete_history_and_sampling(
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
        messages = [
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "First question"),
            ChatMessage("assistant", "First answer"),
            ChatMessage("user", "Follow-up"),
        ]

        result = engine.generate(
            messages,
            max_new_tokens=12,
            sampling=sampling,
            enable_thinking=True,
        )

        self.assertIs(result, generate_text.return_value)
        generate_text.assert_called_once_with(
            model,
            tokenizer,
            messages,
            max_new_tokens=12,
            sampling=sampling,
            enable_thinking=True,
            max_seq_len=256,
            synchronize=engine.synchronize,
        )

class MPSEngineIntegrationTests(unittest.TestCase):
    def _assert_fp16_model_runs_on_mps(self, model_dir: Path) -> None:
        engine = Engine.from_model_dir(
            model_dir, device="mps", dtype="auto", max_seq_len=128
        )

        events = list(
            engine.generate([ChatMessage("user", "Say hello.")], max_new_tokens=2)
        )

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

    @unittest.skipUnless(
        torch.backends.mps.is_available() and QWEN_MODEL_DIR.is_dir(),
        "MPS or local Qwen3 checkpoint is unavailable",
    )
    def test_qwen_fp16_model_inputs_rope_and_cache_run_on_mps(self) -> None:
        self._assert_fp16_model_runs_on_mps(QWEN_MODEL_DIR)

    @unittest.skipUnless(
        torch.backends.mps.is_available() and GRANITE_MODEL_DIR.is_dir(),
        "MPS or local Granite checkpoint is unavailable",
    )
    def test_granite_fp16_model_inputs_rope_and_cache_run_on_mps(self) -> None:
        self._assert_fp16_model_runs_on_mps(GRANITE_MODEL_DIR)

    @unittest.skipUnless(
        torch.backends.mps.is_available() and GRANITE_MODEL_DIR.is_dir(),
        "MPS or local Granite checkpoint is unavailable",
    )
    def test_granite_fp16_greedy_tokens_match_cpu_past_routing_boundary(
        self,
    ) -> None:
        messages = [
            ChatMessage("user", "Explain grouped-query attention briefly.")
        ]

        cpu_engine = Engine.from_model_dir(
            GRANITE_MODEL_DIR,
            device="cpu",
            dtype="float16",
            max_seq_len=256,
        )
        cpu_token_ids = [
            event.token_id
            for event in cpu_engine.generate(messages, max_new_tokens=40)
        ]
        del cpu_engine
        gc.collect()

        mps_engine = Engine.from_model_dir(
            GRANITE_MODEL_DIR,
            device="mps",
            dtype="float16",
            max_seq_len=256,
        )
        mps_token_ids = [
            event.token_id
            for event in mps_engine.generate(messages, max_new_tokens=40)
        ]

        self.assertEqual(mps_token_ids, cpu_token_ids)


if __name__ == "__main__":
    unittest.main()
