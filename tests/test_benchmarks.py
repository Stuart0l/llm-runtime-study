from __future__ import annotations

from argparse import Namespace
import asyncio
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import httpx
import torch

from benchmarks import end_to_end, staggered_http
from benchmarks.__main__ import BenchmarkError, _selected_devices, main, run
from benchmarks.common import PromptCase, measure, render_table
from benchmarks.staggered_http import StaggeredHttpError
from mini_llm.generation import GenerationEvent


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


class EndToEndBenchmarkTests(unittest.TestCase):
    def test_batch_size_controls_requests_and_aggregate_decode_throughput(
        self,
    ) -> None:
        engine = MagicMock()
        engine.device = torch.device("cpu")
        engine.last_cache_num_bytes = 1024**2
        engine.generate.return_value = iter(
            (
                (
                    0,
                    GenerationEvent(
                        2, 0, "", "", model_seconds=0.1, prompt_token_count=3
                    ),
                ),
                (
                    1,
                    GenerationEvent(
                        2, 0, "", "", model_seconds=0.2, prompt_token_count=3
                    ),
                ),
                (
                    0,
                    GenerationEvent(
                        3,
                        1,
                        "",
                        "",
                        model_seconds=0.05,
                        finish_reason="max_new_tokens",
                    ),
                ),
                (
                    1,
                    GenerationEvent(
                        3,
                        1,
                        "",
                        "",
                        model_seconds=0.05,
                        finish_reason="max_new_tokens",
                    ),
                ),
            )
        )

        with patch(
            "benchmarks.end_to_end.time.perf_counter", side_effect=[1.0, 1.1]
        ):
            rows = end_to_end.run(
                engine,
                [PromptCase(3, "prompt", (1, 2, 3))],
                warmups=0,
                repeats=1,
                decode_tokens=2,
                batch_size=2,
            )

        self.assertEqual(len(engine.generate.call_args.args[0]), 2)
        self.assertEqual(rows[0][1:3], ("2", "3"))
        self.assertEqual(rows[0][5:8], ("50.00 ms", "40.00", "4"))
        self.assertEqual(rows[0][-1], "2.00 MiB")


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
            "batch_size": 1,
            "requests": 2,
            "arrival_interval_ms": 0.0,
            "url": None,
            "max_in_flight": None,
            "served_model_name": None,
        }
        values.update(overrides)
        return Namespace(**values)

    @patch("benchmarks.__main__.torch.mps.empty_cache")
    @patch("benchmarks.__main__.gc.collect")
    @patch("benchmarks.__main__.end_to_end.run", return_value=[])
    @patch("benchmarks.__main__.moe_prefill.run", return_value=[])
    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.load_config")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_loads_once_and_moves_same_engine_after_all_cpu_suites(
        self,
        from_model_dir: MagicMock,
        load_config: MagicMock,
        build_prompt_case: MagicMock,
        moe_run: MagicMock,
        end_to_end_run: MagicMock,
        _collect: MagicMock,
        _empty_cache: MagicMock,
    ) -> None:
        load_config.return_value = SimpleNamespace(quantization_config=None)
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
        calls.attach_mock(moe_run, "moe")
        calls.attach_mock(end_to_end_run, "end_to_end")
        calls.attach_mock(engine.to, "move")

        with (
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            run(self._args(), output=StringIO())

        from_model_dir.assert_called_once_with(
            Path("granite"),
            device="cpu",
            dtype="float16",
            max_seq_len=97,
            max_batch_size=1,
        )
        engine.to.assert_called_once_with(device="mps")
        self.assertEqual(moe_run.call_count, 2)
        self.assertEqual(end_to_end_run.call_count, 2)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            [
                "moe",
                "end_to_end",
                "move",
                "moe",
                "end_to_end",
            ],
        )
        for suite in (moe_run, end_to_end_run):
            self.assertIs(suite.call_args_list[0].args[0], engine)
            self.assertIs(suite.call_args_list[1].args[0], engine)
        for call in end_to_end_run.call_args_list:
            self.assertEqual(call.kwargs["batch_size"], 1)

    def test_default_devices_include_available_accelerators(self) -> None:
        cases = (
            (True, False, ["cpu", "mps"]),
            (False, True, ["cpu", "cuda"]),
        )
        for mps_available, cuda_available, expected in cases:
            with (
                self.subTest(mps=mps_available, cuda=cuda_available),
                patch("torch.backends.mps.is_available", return_value=mps_available),
                patch("torch.cuda.is_available", return_value=cuda_available),
            ):
                self.assertEqual(_selected_devices(None), expected)

    def test_explicit_unavailable_device_fails(self) -> None:
        cases = (
            ("mps", "torch.backends.mps.is_available"),
            ("cuda", "torch.cuda.is_available"),
        )
        for device, availability in cases:
            with (
                self.subTest(device=device),
                patch(availability, return_value=False),
                self.assertRaisesRegex(BenchmarkError, "unavailable"),
            ):
                _selected_devices([device])

    @patch("benchmarks.__main__.end_to_end.run", return_value=[])
    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    @patch("benchmarks.__main__.load_config")
    def test_default_devices_select_cuda_only_for_gptq(
        self,
        load_config: MagicMock,
        from_model_dir: MagicMock,
        build_prompt_case: MagicMock,
        _end_to_end: MagicMock,
    ) -> None:
        load_config.return_value = SimpleNamespace(quantization_config=object())
        engine = MagicMock()
        engine.max_seq_len = 97
        engine.load_seconds = 1.0
        engine.model.config.model_type = "qwen3"
        from_model_dir.return_value = engine
        build_prompt_case.return_value = PromptCase(32, "prompt", (1, 2))

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            run(
                self._args(model=[Path("qwen-gptq")], benchmark=["end-to-end"]),
                output=StringIO(),
            )

        from_model_dir.assert_called_once_with(
            Path("qwen-gptq"),
            device="cuda",
            dtype="float16",
            max_seq_len=97,
            max_batch_size=1,
        )
        engine.to.assert_not_called()

    @patch("benchmarks.__main__.Engine.from_model_dir")
    @patch("benchmarks.__main__.load_config")
    def test_rejects_explicit_cpu_before_loading_gptq(
        self, load_config: MagicMock, from_model_dir: MagicMock
    ) -> None:
        load_config.return_value = SimpleNamespace(quantization_config=object())

        with self.assertRaisesRegex(BenchmarkError, "CUDA only"):
            run(
                self._args(
                    model=[Path("qwen-gptq")],
                    benchmark=["end-to-end"],
                    device=["cpu"],
                ),
                output=StringIO(),
            )

        from_model_dir.assert_not_called()

    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.load_config")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_explicit_moe_benchmark_rejects_qwen(
        self,
        from_model_dir: MagicMock,
        load_config: MagicMock,
        build_prompt_case: MagicMock,
    ) -> None:
        load_config.return_value = SimpleNamespace(quantization_config=None)
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


    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_staggered_http_rejects_invalid_combinations_without_loading(
        self, from_model_dir: MagicMock
    ) -> None:
        complete = {"url": "http://server/v1", "max_in_flight": 2}
        cases = {
            "another suite": self._args(
                benchmark=["staggered-http", "staggered"], **complete
            ),
            "device": self._args(
                benchmark=["staggered-http"], device=["cuda"], **complete
            ),
            "two models": self._args(
                benchmark=["staggered-http"],
                model=[Path("a"), Path("b")],
                **complete,
            ),
            "missing url": self._args(benchmark=["staggered-http"], max_in_flight=2),
            "missing max in-flight": self._args(
                benchmark=["staggered-http"], url="http://server/v1"
            ),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(BenchmarkError):
                run(args, output=StringIO())
        from_model_dir.assert_not_called()

    @patch("benchmarks.__main__.staggered_http.run", return_value=[])
    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.load_tokenizer")
    @patch("benchmarks.__main__.Engine.from_model_dir")
    def test_staggered_http_loads_only_the_tokenizer(
        self,
        from_model_dir: MagicMock,
        load_tokenizer: MagicMock,
        build_prompt_case: MagicMock,
        http_run: MagicMock,
    ) -> None:
        build_prompt_case.return_value = PromptCase(32, "prompt", (1, 2))

        run(
            self._args(
                model=[Path("models/qwen3-0.6b")],
                benchmark=["staggered-http"],
                url="http://server/v1",
                max_in_flight=4,
                served_model_name="served",
            ),
            output=StringIO(),
        )

        from_model_dir.assert_not_called()
        load_tokenizer.assert_called_once_with(Path("models/qwen3-0.6b"))
        self.assertIs(build_prompt_case.call_args.args[0], load_tokenizer.return_value)
        self.assertEqual(http_run.call_args.kwargs["served_model"], "served")
        self.assertEqual(http_run.call_args.kwargs["max_in_flight"], 4)


def _completion(prompt_tokens: int = 2, completion_tokens: int = 3) -> dict:
    return {
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }


class StaggeredHttpTests(unittest.TestCase):
    case = PromptCase(2, "prompt", (1, 2))

    def _run(self, transport: httpx.AsyncBaseTransport, **overrides) -> list:
        options = {
            "url": "http://server/v1",
            "served_model": "served",
            "max_in_flight": 2,
            "warmups": 0,
            "repeats": 1,
            "decode_tokens": 3,
            "requests": 5,
            "interval_seconds": 0.0,
            "transport": transport,
        }
        options.update(overrides)
        return staggered_http.run([self.case], **options)

    def test_never_exceeds_max_in_flight_and_reports_one_row(self) -> None:
        active = 0
        peak = 0
        bodies = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            bodies.append(json.loads(request.content))
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json=_completion())

        rows = self._run(httpx.MockTransport(handler))

        self.assertEqual(peak, 2)
        self.assertEqual(len(bodies), 5)
        self.assertEqual(
            bodies[0],
            {
                "model": "served",
                "messages": [{"role": "user", "content": "prompt"}],
                "temperature": 0,
                "max_completion_tokens": 3,
            },
        )
        self.assertEqual(
            rows[0][:5], ("http://server/v1", "5", "0 ms", "2", "2")
        )

    def test_latency_includes_time_waiting_for_a_client_slot(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=_completion())

        rows = self._run(
            httpx.MockTransport(handler), max_in_flight=1, requests=2
        )

        # The second request waits ~50 ms for the only slot, then takes ~50 ms,
        # so p95 latency (the slower request) is about 100 ms, not 50 ms.
        p95 = float(rows[0][8].removesuffix(" ms"))
        self.assertGreater(p95, 90.0)

    def test_failed_responses_raise_staggered_http_error(self) -> None:
        def connect_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        handlers = {
            "429": lambda _request: httpx.Response(429, json={"error": {}}),
            "missing usage": lambda _request: httpx.Response(200, json={}),
            "prompt mismatch": lambda _request: httpx.Response(
                200, json=_completion(prompt_tokens=9)
            ),
            "connection error": connect_error,
        }
        for name, handler in handlers.items():
            with self.subTest(name), self.assertRaises(StaggeredHttpError):
                self._run(httpx.MockTransport(handler))

    @patch("benchmarks.__main__.build_prompt_case")
    @patch("benchmarks.__main__.load_tokenizer")
    def test_connection_error_is_reported_by_main(
        self, _load_tokenizer: MagicMock, build_prompt_case: MagicMock
    ) -> None:
        build_prompt_case.return_value = self.case
        error = StringIO()

        # Nothing listens on port 9 (discard); the request fails to connect.
        status = main(
            [
                "--model",
                "models/qwen3-0.6b",
                "--benchmark",
                "staggered-http",
                "--url",
                "http://127.0.0.1:9/v1",
                "--max-in-flight",
                "1",
                "--requests",
                "1",
                "--prompt-lengths",
                "2",
                "--repeats",
                "1",
                "--warmups",
                "0",
            ],
            output=StringIO(),
            error=error,
        )

        self.assertEqual(status, 2)
        self.assertTrue(error.getvalue().startswith("error: request 0:"))


if __name__ == "__main__":
    unittest.main()
