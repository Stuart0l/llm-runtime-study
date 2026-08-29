from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import torch

from benchmarks import cache_decode
from benchmarks.__main__ import BenchmarkError, _selected_devices, run
from benchmarks.common import PromptCase, measure, render_table


class MeasurementTests(unittest.TestCase):
    def test_measure_excludes_warmup_and_preparation(self) -> None:
        operation = MagicMock(side_effect=["warmup", "first", "second"])
        prepare = MagicMock()
        synchronize = MagicMock()

        with patch(
            "benchmarks.common.time.perf_counter",
            side_effect=[1.0, 1.1, 2.0, 2.25],
        ):
            measured = measure(
                operation,
                synchronize=synchronize,
                warmups=1,
                repeats=2,
                prepare=prepare,
            )

        self.assertEqual(measured.results, ("first", "second"))
        self.assertAlmostEqual(measured.seconds[0], 0.1)
        self.assertAlmostEqual(measured.seconds[1], 0.25)
        self.assertAlmostEqual(measured.median_seconds, 0.175)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(prepare.call_count, 3)
        self.assertEqual(synchronize.call_count, 6)

    def test_render_table_aligns_columns(self) -> None:
        rendered = render_table(("name", "value"), (("short", 1), ("longer", 20)))

        self.assertEqual(
            rendered,
            "name    value\n------  -----\nshort   1    \nlonger  20   ",
        )

    def test_measure_validates_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmups"):
            measure(lambda: None, synchronize=lambda: None, warmups=-1, repeats=1)
        with self.assertRaisesRegex(ValueError, "repeats"):
            measure(lambda: None, synchronize=lambda: None, warmups=0, repeats=0)


class CacheDecodeBenchmarkTests(unittest.TestCase):
    def test_long_prompt_skips_entire_comparison(self) -> None:
        logits = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        model = MagicMock()
        model.prefill.return_value = logits
        model.decode.return_value = logits
        model.cache = SimpleNamespace(num_bytes=1024)
        engine = SimpleNamespace(
            model=model,
            device=torch.device("cpu"),
            synchronize=lambda: None,
        )

        rows = cache_decode.run(
            engine,
            [PromptCase(128, "prompt", (1, 2))],
            warmups=0,
            repeats=1,
            decode_tokens=1,
        )

        model.assert_not_called()
        self.assertEqual(rows, [])

    def test_table_names_cached_speedup_explicitly(self) -> None:
        self.assertIn("cached speedup", cache_decode.HEADERS)


class RunnerTests(unittest.TestCase):
    def _args(self, **overrides: object) -> Namespace:
        values = {
            "model": [Path("granite")],
            "benchmark": None,
            "device": None,
            "prompt_lengths": [32],
            "warmups": 0,
            "repeats": 1,
            "decode_tokens": 1,
        }
        values.update(overrides)
        return Namespace(**values)

    @patch("benchmarks.__main__.torch.mps.empty_cache")
    @patch("benchmarks.__main__.gc.collect")
    @patch("benchmarks.__main__.end_to_end.run", return_value=[])
    @patch("benchmarks.__main__.moe_prefill.run", return_value=[])
    @patch("benchmarks.__main__.cache_decode.run", return_value=[])
    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_loads_once_and_moves_same_engine_after_all_cpu_suites(
        self,
        from_model_dir: MagicMock,
        build_prompt_case: MagicMock,
        cache_run: MagicMock,
        moe_run: MagicMock,
        end_to_end_run: MagicMock,
        _collect: MagicMock,
        _empty_cache: MagicMock,
    ) -> None:
        engine = MagicMock()
        engine.device = torch.device("cpu")
        engine.dtype = torch.float16
        engine.max_seq_len = 97
        engine.load_seconds = 1.0
        engine.model.config.model_type = "granitemoe"

        def move(*, device: str) -> object:
            engine.device = torch.device(device)
            return engine

        engine.to.side_effect = move
        from_model_dir.return_value = engine
        case = PromptCase(32, "prompt", (1, 2))
        build_prompt_case.return_value = case
        calls = MagicMock()
        calls.attach_mock(cache_run, "cache")
        calls.attach_mock(moe_run, "moe")
        calls.attach_mock(end_to_end_run, "end_to_end")
        calls.attach_mock(engine.to, "move")

        with patch("torch.backends.mps.is_available", return_value=True):
            run(self._args(), output=StringIO())

        from_model_dir.assert_called_once_with(
            Path("granite"),
            device="cpu",
            dtype="float16",
            max_seq_len=97,
        )
        engine.to.assert_called_once_with(device="mps")
        self.assertEqual(cache_run.call_count, 2)
        self.assertEqual(moe_run.call_count, 2)
        self.assertEqual(end_to_end_run.call_count, 2)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            [
                "cache",
                "moe",
                "end_to_end",
                "move",
                "cache",
                "moe",
                "end_to_end",
            ],
        )
        for suite in (cache_run, moe_run, end_to_end_run):
            self.assertIs(suite.call_args_list[0].args[0], engine)
            self.assertIs(suite.call_args_list[1].args[0], engine)

    def test_default_devices_include_available_mps(self) -> None:
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(_selected_devices(None), ["cpu", "mps"])

    def test_explicit_unavailable_mps_fails(self) -> None:
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(BenchmarkError, "unavailable"),
        ):
            _selected_devices(["mps"])

    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_explicit_moe_benchmark_rejects_qwen(
        self, from_model_dir: MagicMock, build_prompt_case: MagicMock
    ) -> None:
        engine = MagicMock()
        engine.max_seq_len = 97
        engine.load_seconds = 1.0
        engine.model.config.model_type = "qwen3"
        from_model_dir.return_value = engine
        build_prompt_case.return_value = PromptCase(32, "prompt", (1, 2))

        with self.assertRaisesRegex(BenchmarkError, "requires Granite"):
            run(
                self._args(benchmark=["moe-prefill"], device=["cpu"]),
                output=StringIO(),
            )


if __name__ == "__main__":
    unittest.main()
