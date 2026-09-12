from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from tests.reference_support import has_local_checkpoint

from mini_llm.cache.paged import SequenceCacheHandle
from mini_llm.config import GraniteMoeConfig, Qwen3Config
from mini_llm.engine import (
    Engine,
    EngineError,
    resolve_device,
    resolve_dtype,
    infer_quantization,
    synchronize_device,
)
from mini_llm.model.base import CausalLMBase
from mini_llm.sampling import SamplingConfig
from mini_llm.tokenizer import ChatMessage


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


class DeviceAndDtypeTests(unittest.TestCase):
    def test_auto_prefers_cuda_over_mps(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cuda"))

    def test_auto_prefers_mps_when_cuda_is_unavailable(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("mps"))

    def test_auto_falls_back_to_cpu(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))

    def test_explicit_unavailable_mps_has_clear_error(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(EngineError, "MPS.*unavailable"),
        ):
            resolve_device("mps")

    def test_explicit_unavailable_cuda_has_clear_error(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(EngineError, "CUDA.*unavailable"),
        ):
            resolve_device("cuda")

    def test_cuda_device_index_is_validated(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            self.assertEqual(resolve_device("cuda:0"), torch.device("cuda:0"))
            with self.assertRaisesRegex(EngineError, "index 1.*unavailable"):
                resolve_device("cuda:1")

    def test_rejects_unsupported_device_type(self) -> None:
        with self.assertRaisesRegex(EngineError, "supports cpu, mps, and cuda"):
            resolve_device("xpu")

    def test_auto_dtype_is_fp16_on_accelerators_and_fp32_on_cpu(self) -> None:
        self.assertEqual(
            resolve_dtype("auto", device=torch.device("cuda")), torch.float16
        )
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

    def test_cuda_synchronization_targets_selected_device(self) -> None:
        with patch("torch.cuda.synchronize") as synchronize:
            synchronize_device(torch.device("cuda:0"))

        synchronize.assert_called_once_with(torch.device("cuda:0"))


class QuantizationDetectionTests(unittest.TestCase):
    def _config(self, *, quantized: bool) -> MagicMock:
        config = MagicMock(spec=Qwen3Config)
        config.quantization_config = MagicMock() if quantized else None
        return config

    def test_infers_mode_from_checkpoint_metadata(self) -> None:
        self.assertEqual(
            infer_quantization(self._config(quantized=False)),
            "dense",
        )
        self.assertEqual(
            infer_quantization(self._config(quantized=True)),
            "gptq-marlin",
        )



class EngineTests(unittest.TestCase):
    def _mock_engine(self) -> Engine:
        return Engine(
            model=MagicMock(),
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_seq_len=256,
            load_seconds=1.0,
        )

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
        model.input_device = torch.device("cpu")
        model.parameters.return_value = iter(
            [MagicMock(dtype=torch.bfloat16)]
        )
        load_model.return_value = model
        tokenizer = MagicMock()
        load_tokenizer.return_value = tokenizer
        config = MagicMock(spec=Qwen3Config)
        config.quantization_config = None
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
        self.assertEqual(engine.quantization, "dense")
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
        config.quantization_config = None
        load_typed_config.return_value = config
        model = MagicMock()
        model.to.return_value = model
        model.input_device = torch.device("cpu")
        model.parameters.return_value = iter(
            [MagicMock(dtype=torch.bfloat16)]
        )
        load_model.return_value = model

        engine = Engine.from_model_dir(
            "granite", device="cpu", dtype="bfloat16", max_seq_len=128
        )

        self.assertIs(engine.model, model)
        load_tokenizer.assert_called_once_with("granite", model_config=config)
        load_model.assert_called_once_with("granite", model_config=config)

    @patch("mini_llm.engine.PagedKVCachePool")
    def test_paged_cache_is_the_default_runtime_backend(
        self, paged_pool: MagicMock
    ) -> None:
        manager = MagicMock()
        paged_pool.return_value = manager
        engine = self._mock_engine()

        self.assertIs(engine.cache_manager, manager)
        paged_pool.assert_called_once_with(
            engine.model.config,
            256,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    @patch("mini_llm.engine.DenseKVCacheManager")
    def test_dense_cache_backend_is_runtime_selectable(
        self, dense_manager: MagicMock
    ) -> None:
        manager = MagicMock()
        dense_manager.return_value = manager
        engine = self._mock_engine()
        engine.cache_backend = "dense"

        self.assertIs(engine.cache_manager, manager)
        dense_manager.assert_called_once_with(
            engine.model.config,
            256,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    @patch("mini_llm.engine.PagedDecodeGraph")
    def test_cuda_graph_decode_can_be_disabled(
        self, paged_decode_graph: MagicMock
    ) -> None:
        model = MagicMock(spec=CausalLMBase)
        engine = Engine(
            model=model,
            tokenizer=MagicMock(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            max_seq_len=16,
            load_seconds=0.0,
            use_cuda_graph=False,
        )
        cache = MagicMock(spec=SequenceCacheHandle)
        input_ids = torch.tensor([[1]])

        engine._make_decode(cache)(input_ids)

        paged_decode_graph.assert_not_called()
        model.decode.assert_called_once_with(input_ids, cache=cache)

    @patch("mini_llm.engine.PagedDecodeGraph")
    def test_reuses_decode_graph_across_block_counts(
        self, paged_decode_graph: MagicMock
    ) -> None:
        model = MagicMock(spec=CausalLMBase)
        engine = Engine(
            model=model,
            tokenizer=MagicMock(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            max_seq_len=32,
            load_seconds=0.0,
        )
        pool = MagicMock()
        first_cache = MagicMock(spec=SequenceCacheHandle)
        first_cache.pool = pool
        first_cache.block_table = torch.tensor([0, 1])
        second_cache = MagicMock(spec=SequenceCacheHandle)
        second_cache.pool = pool
        second_cache.block_table = torch.tensor([2, 3, 4])
        paged_decode_graph.return_value.cache.pool = pool

        first_decode = engine._make_decode(first_cache)
        second_decode = engine._make_decode(second_cache)

        paged_decode_graph.assert_called_once_with(model, first_cache)
        paged_decode_graph.return_value.cache.bind.assert_called_once_with(
            second_cache
        )
        self.assertIs(first_decode, paged_decode_graph.return_value.replay)
        self.assertIs(second_decode, paged_decode_graph.return_value.replay)

    def test_rejects_unknown_cache_backend(self) -> None:
        with self.assertRaisesRegex(EngineError, "cache backend"):
            Engine(
                model=MagicMock(),
                tokenizer=MagicMock(),
                device=torch.device("cpu"),
                dtype=torch.float32,
                max_seq_len=16,
                load_seconds=0.0,
                cache_backend="unknown",  # type: ignore[arg-type]
            )

    def test_to_rejects_moving_with_an_active_request_cache(self) -> None:
        engine = self._mock_engine()
        manager = MagicMock()
        manager.active_sequences = 1
        engine._cache_manager = manager

        with self.assertRaisesRegex(EngineError, "request caches are active"):
            engine.to(dtype="float16")

        engine.model.to.assert_not_called()

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
        cache_manager = MagicMock()
        engine._cache_manager = cache_manager
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
            cache_manager=cache_manager,
            decode_factory=engine._make_decode,
        )

    @patch("mini_llm.engine.synchronize_device")
    def test_to_moves_resident_model_and_updates_placement(
        self, synchronize: MagicMock
    ) -> None:
        engine = self._mock_engine()
        engine._decode_graph = MagicMock()

        result = engine.to(device="cpu", dtype="float16")

        self.assertIs(result, engine)
        engine.model.to.assert_called_once_with(
            device=torch.device("cpu"), dtype=torch.float16
        )
        engine.model.materialize_derived_buffers.assert_called_once_with(
            torch.device("cpu")
        )
        synchronize.assert_called_once_with(torch.device("cpu"))
        self.assertEqual(engine.device, torch.device("cpu"))
        self.assertEqual(engine.dtype, torch.float16)
        self.assertIsNone(engine._decode_graph)

    @patch("mini_llm.engine.synchronize_device")
    def test_to_retains_omitted_dtype_when_changing_device(
        self, synchronize: MagicMock
    ) -> None:
        engine = self._mock_engine()
        with patch("torch.backends.mps.is_available", return_value=True):
            engine.to(device="mps")

        engine.model.to.assert_called_once_with(
            device=torch.device("mps"), dtype=torch.float32
        )
        self.assertEqual(engine.device, torch.device("mps"))
        self.assertEqual(engine.dtype, torch.float32)
        synchronize.assert_called_once_with(torch.device("mps"))

    @patch("mini_llm.engine.synchronize_device")
    def test_to_resolves_auto_dtype_for_destination(
        self, synchronize: MagicMock
    ) -> None:
        engine = self._mock_engine()
        with patch("torch.backends.mps.is_available", return_value=True):
            engine.to(device="mps", dtype="auto")

        self.assertEqual(engine.dtype, torch.float16)
        engine.model.to.assert_called_once_with(
            device=torch.device("mps"), dtype=torch.float16
        )
        synchronize.assert_called_once_with(torch.device("mps"))

    @patch("mini_llm.engine.synchronize_device")
    def test_to_is_noop_for_existing_placement(
        self, synchronize: MagicMock
    ) -> None:
        engine = self._mock_engine()

        result = engine.to()

        self.assertIs(result, engine)
        engine.model.to.assert_not_called()
        engine.model.materialize_derived_buffers.assert_not_called()
        synchronize.assert_not_called()

    @patch("mini_llm.engine.load_model")
    @patch("mini_llm.engine.load_tokenizer")
    @patch("mini_llm.engine.load_config")
    @patch("mini_llm.engine.synchronize_device")
    def test_to_never_reloads_model_or_tokenizer(
        self,
        _synchronize: MagicMock,
        load_config: MagicMock,
        load_tokenizer: MagicMock,
        load_model: MagicMock,
    ) -> None:
        engine = self._mock_engine()

        engine.to(dtype="float16")

        load_config.assert_not_called()
        load_tokenizer.assert_not_called()
        load_model.assert_not_called()

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
        self.assertEqual(engine.cache_manager.active_sequences, 0)
        self.assertEqual(engine.cache_manager.device.type, "mps")

    @unittest.skipUnless(
        torch.backends.mps.is_available() and has_local_checkpoint(QWEN_MODEL_DIR),
        "MPS or local Qwen3 checkpoint is unavailable",
    )
    def test_qwen_fp16_model_inputs_rope_and_cache_run_on_mps(self) -> None:
        self._assert_fp16_model_runs_on_mps(QWEN_MODEL_DIR)

    @unittest.skipUnless(
        torch.backends.mps.is_available() and has_local_checkpoint(GRANITE_MODEL_DIR),
        "MPS or local Granite checkpoint is unavailable",
    )
    def test_granite_fp16_model_inputs_rope_and_cache_run_on_mps(self) -> None:
        self._assert_fp16_model_runs_on_mps(GRANITE_MODEL_DIR)

    @unittest.skipUnless(
        torch.backends.mps.is_available() and has_local_checkpoint(GRANITE_MODEL_DIR),
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

    @unittest.skipUnless(
        torch.backends.mps.is_available() and has_local_checkpoint(QWEN_MODEL_DIR),
        "MPS or local Qwen3 checkpoint is unavailable",
    )
    def test_loaded_cpu_engine_moves_to_mps_without_reloading(self) -> None:
        engine = Engine.from_model_dir(
            QWEN_MODEL_DIR, device="cpu", dtype="float16", max_seq_len=128
        )
        list(engine.generate([ChatMessage("user", "Hello")], max_new_tokens=1))
        old_manager = engine.cache_manager
        self.assertEqual(old_manager.active_sequences, 0)

        result = engine.to(device="mps")

        self.assertIs(result, engine)
        self.assertIsNone(engine._cache_manager)
        self.assertEqual(engine.device, torch.device("mps"))
        self.assertEqual(engine.dtype, torch.float16)
        self.assertEqual(engine.model.input_device.type, "mps")
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.dtype,
            torch.float32,
        )
        events = list(
            engine.generate([ChatMessage("user", "Hello")], max_new_tokens=1)
        )
        self.assertTrue(events)


class CUDAEngineIntegrationTests(unittest.TestCase):
    def _assert_fp16_model_runs_on_cuda(self, model_dir: Path) -> None:
        engine = Engine.from_model_dir(
            model_dir, device="cuda", dtype="auto", max_seq_len=128
        )

        events = list(
            engine.generate([ChatMessage("user", "Say hello.")], max_new_tokens=2)
        )

        self.assertEqual(engine.device, torch.device("cuda"))
        self.assertEqual(engine.dtype, torch.float16)
        self.assertTrue(events)
        self.assertEqual(engine.model.model.embed_tokens.weight.device.type, "cuda")
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.device.type, "cuda"
        )
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.dtype, torch.float32
        )
        self.assertEqual(engine.cache_manager.active_sequences, 0)
        self.assertEqual(engine.cache_manager.device.type, "cuda")

    @unittest.skipUnless(
        torch.cuda.is_available() and has_local_checkpoint(QWEN_MODEL_DIR),
        "CUDA or local Qwen3 checkpoint is unavailable",
    )
    def test_qwen_fp16_model_inputs_rope_and_cache_run_on_cuda(self) -> None:
        self._assert_fp16_model_runs_on_cuda(QWEN_MODEL_DIR)

    @unittest.skipUnless(
        torch.cuda.is_available() and has_local_checkpoint(GRANITE_MODEL_DIR),
        "CUDA or local Granite checkpoint is unavailable",
    )
    def test_granite_fp16_model_inputs_rope_and_cache_run_on_cuda(self) -> None:
        self._assert_fp16_model_runs_on_cuda(GRANITE_MODEL_DIR)

    @unittest.skipUnless(
        torch.cuda.is_available() and has_local_checkpoint(QWEN_MODEL_DIR),
        "CUDA or local Qwen3 checkpoint is unavailable",
    )
    def test_loaded_cpu_engine_moves_to_cuda_without_reloading(self) -> None:
        engine = Engine.from_model_dir(
            QWEN_MODEL_DIR, device="cpu", dtype="float16", max_seq_len=128
        )
        list(engine.generate([ChatMessage("user", "Hello")], max_new_tokens=1))
        old_manager = engine.cache_manager
        self.assertEqual(old_manager.active_sequences, 0)

        result = engine.to(device="cuda")

        self.assertIs(result, engine)
        self.assertIsNone(engine._cache_manager)
        self.assertEqual(engine.device, torch.device("cuda"))
        self.assertEqual(engine.dtype, torch.float16)
        self.assertEqual(engine.model.input_device.type, "cuda")
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.dtype,
            torch.float32,
        )
        events = list(
            engine.generate([ChatMessage("user", "Hello")], max_new_tokens=1)
        )
        self.assertTrue(events)


if __name__ == "__main__":
    unittest.main()
