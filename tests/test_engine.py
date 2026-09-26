from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest

import torch

from tests.reference_support import has_local_checkpoint
from tests.test_qwen_model import _tiny_config

from mini_llm.cache.dense import DenseKVCacheManager
from mini_llm.cache.paged import PagedKVCacheManager, PagedSequenceKVCache
from mini_llm.engine import Engine, EngineError, resolve_device, resolve_dtype
from mini_llm.model.base import CausalLMBase
from mini_llm.model.qwen import Qwen3ForCausalLM
from mini_llm.tokenizer import ChatMessage


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


def _assert_moves_without_reloading(test: unittest.TestCase, device: str) -> None:
    engine = Engine.from_model_dir(
        QWEN_MODEL_DIR, device="cpu", dtype="float16", max_seq_len=128
    )
    list(engine.generate(([ChatMessage("user", "Hello")],), max_new_tokens=1))
    old_manager = engine.cache_manager
    test.assertEqual(old_manager.active_sequences, 0)

    result = engine.to(device=device)

    test.assertIs(result, engine)
    test.assertEqual(engine.cache_manager.spec.device.type, device)
    test.assertEqual(engine.device, torch.device(device))
    test.assertEqual(engine.dtype, torch.float16)
    test.assertEqual(engine.model.input_device.type, device)
    test.assertEqual(
        engine.model.model.rotary_emb.inverse_frequencies.dtype,
        torch.float32,
    )
    events = list(
        engine.generate(([ChatMessage("user", "Hello")],), max_new_tokens=1)
    )
    test.assertTrue(events)


class DeviceAndDtypeTests(unittest.TestCase):
    def test_auto_device_precedence(self) -> None:
        cases = (
            (True, True, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        )
        for cuda_available, mps_available, expected in cases:
            with (
                self.subTest(
                    cuda_available=cuda_available, mps_available=mps_available
                ),
                patch("torch.cuda.is_available", return_value=cuda_available),
                patch("torch.backends.mps.is_available", return_value=mps_available),
            ):
                self.assertEqual(resolve_device("auto"), torch.device(expected))

    def test_explicit_unavailable_device_has_clear_error(self) -> None:
        cases = (
            ("mps", "MPS.*unavailable"),
            ("cuda", "CUDA.*unavailable"),
        )
        for device, message in cases:
            with (
                self.subTest(device=device),
                patch("torch.cuda.is_available", return_value=False),
                patch("torch.backends.mps.is_available", return_value=False),
                self.assertRaisesRegex(EngineError, message),
            ):
                resolve_device(device)

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

    def test_cache_manager_follows_backend_and_batch_size(self) -> None:
        cases = (
            ("paged", {"cache_backend": "paged"}, PagedKVCacheManager),
            ("dense", {"cache_backend": "dense"}, DenseKVCacheManager),
            ("default", {}, PagedKVCacheManager),
        )
        for name, backend_kwargs, expected_type in cases:
            with self.subTest(backend=name):
                engine = Engine(
                    model=Qwen3ForCausalLM(_tiny_config()),
                    tokenizer=MagicMock(),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    max_seq_len=16,
                    load_seconds=0.0,
                    max_batch_size=3,
                    **backend_kwargs,
                )

                self.assertIsInstance(engine.cache_manager, expected_type)
                self.assertEqual(engine.cache_manager.capacity, 48)

    @patch("mini_llm.engine.PagedDecodeGraphCache")
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
        cache = MagicMock(spec=PagedSequenceKVCache)
        input_ids = torch.tensor([[1]])

        engine._make_decode((cache,))(input_ids)

        paged_decode_graph.assert_not_called()
        model.decode.assert_called_once_with(input_ids, caches=(cache,))

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

    def test_rejects_request_batch_larger_than_configured_maximum(self) -> None:
        engine = self._mock_engine()

        with self.assertRaisesRegex(EngineError, "exceeds engine maximum"):
            engine.generate(
                (
                    [ChatMessage("user", "first")],
                    [ChatMessage("user", "second")],
                ),
                max_new_tokens=1,
            )

    def test_to_rejects_moving_with_an_active_request_cache(self) -> None:
        engine = self._mock_engine()
        manager = MagicMock()
        manager.active_sequences = 1
        engine._cache_manager = manager

        with self.assertRaisesRegex(EngineError, "request caches are active"):
            engine.to(dtype="float16")

        engine.model.to.assert_not_called()

    def test_to_moves_resident_model_and_updates_placement(self) -> None:
        engine = Engine(
            model=Qwen3ForCausalLM(_tiny_config()),
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_seq_len=16,
            load_seconds=0.0,
        )
        _ = engine.cache_manager
        engine._decode_graphs = object()  # type: ignore[assignment]

        result = engine.to(device="cpu", dtype="float16")

        self.assertIs(result, engine)
        for name, parameter in engine.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertEqual(parameter.dtype, torch.float16)
        self.assertEqual(
            engine.model.model.rotary_emb.inverse_frequencies.dtype, torch.float32
        )
        self.assertEqual(engine.dtype, torch.float16)
        self.assertIsNone(engine._decode_graphs)
        self.assertEqual(engine.cache_manager.spec.dtype, torch.float16)

    def test_to_dtype_precedence(self) -> None:
        cases = (
            (None, torch.float32),
            ("auto", torch.float16),
        )
        for requested_dtype, expected_dtype in cases:
            with (
                self.subTest(dtype=requested_dtype),
                patch("mini_llm.engine.synchronize_device") as synchronize,
                patch("torch.backends.mps.is_available", return_value=True),
            ):
                engine = self._mock_engine()

                engine.to(device="mps", dtype=requested_dtype)

                engine.model.to.assert_called_once_with(
                    device=torch.device("mps"), dtype=expected_dtype
                )
                self.assertEqual(engine.device, torch.device("mps"))
                self.assertEqual(engine.dtype, expected_dtype)
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


class MPSEngineIntegrationTests(unittest.TestCase):
    def _assert_fp16_model_runs_on_mps(self, model_dir: Path) -> None:
        engine = Engine.from_model_dir(
            model_dir, device="mps", dtype="auto", max_seq_len=128
        )

        events = list(
            engine.generate(([ChatMessage("user", "Say hello.")],), max_new_tokens=2)
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
        self.assertEqual(engine.cache_manager.spec.device.type, "mps")

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
            for _, event in cpu_engine.generate((messages,), max_new_tokens=40)
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
            for _, event in mps_engine.generate((messages,), max_new_tokens=40)
        ]

        self.assertEqual(mps_token_ids, cpu_token_ids)

    @unittest.skipUnless(
        torch.backends.mps.is_available() and has_local_checkpoint(QWEN_MODEL_DIR),
        "MPS or local Qwen3 checkpoint is unavailable",
    )
    def test_loaded_cpu_engine_moves_to_mps_without_reloading(self) -> None:
        _assert_moves_without_reloading(self, "mps")


class CUDAEngineIntegrationTests(unittest.TestCase):
    def _assert_fp16_model_runs_on_cuda(self, model_dir: Path) -> None:
        engine = Engine.from_model_dir(
            model_dir, device="cuda", dtype="auto", max_seq_len=128
        )

        events = list(
            engine.generate(([ChatMessage("user", "Say hello.")],), max_new_tokens=2)
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
        self.assertEqual(engine.cache_manager.spec.device.type, "cuda")

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
        _assert_moves_without_reloading(self, "cuda")


if __name__ == "__main__":
    unittest.main()
