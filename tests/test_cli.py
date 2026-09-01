from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch
import unittest

import torch

from tests.reference_support import has_local_checkpoint

from mini_llm.cli import build_parser, main
from mini_llm.generation import GenerationEvent
from mini_llm.tokenizer import ChatMessage


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


class CLITests(unittest.TestCase):
    def test_parser_accepts_cuda_device(self) -> None:
        args = build_parser().parse_args(
            ["--model", "model", "--prompt", "hello", "--device", "cuda"]
        )

        self.assertEqual(args.device, "cuda")

    @patch("mini_llm.cli.Engine.from_model_dir")
    def test_interactive_mode_loads_once_and_generates_for_each_prompt(
        self, from_model_dir: MagicMock
    ) -> None:
        engine = MagicMock()
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
        engine.load_seconds = 1.5
        engine.generate.side_effect = [
            iter(
                [
                    GenerationEvent(
                        2,
                        0,
                        "First answer",
                        "First answer",
                        finish_reason="eos",
                        model_seconds=0.1,
                        prompt_token_count=3,
                    )
                ]
            ),
            iter(
                [
                    GenerationEvent(
                        3,
                        0,
                        "Second answer",
                        "Second answer",
                        finish_reason="eos",
                        model_seconds=0.1,
                        prompt_token_count=4,
                    )
                ]
            ),
        ]
        from_model_dir.return_value = engine
        output = StringIO()
        error = StringIO()

        status = main(
            ["--model", "model", "--interactive", "--no-metrics"],
            input_stream=StringIO("first prompt\nsecond prompt\n/quit\n"),
            output=output,
            error=error,
        )

        self.assertEqual(status, 0)
        self.assertEqual(error.getvalue(), "")
        from_model_dir.assert_called_once()
        self.assertEqual(engine.generate.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in engine.generate.call_args_list],
            [
                [ChatMessage("user", "first prompt")],
                [ChatMessage("user", "second prompt")],
            ],
        )
        rendered = output.getvalue()
        self.assertEqual(rendered.count("Loaded model"), 1)
        self.assertIn("assistant> First answer", rendered)
        self.assertIn("assistant> Second answer", rendered)

    @patch("mini_llm.cli.Engine.from_model_dir")
    def test_one_shot_mode_requires_prompt(
        self, from_model_dir: MagicMock
    ) -> None:
        error = StringIO()

        status = main(["--model", "model"], output=StringIO(), error=error)

        self.assertEqual(status, 2)
        self.assertIn("--prompt is required", error.getvalue())
        from_model_dir.assert_not_called()

    @patch("mini_llm.cli.Engine.from_model_dir")
    def test_streams_text_and_prints_benchmark_summary(
        self, from_model_dir: MagicMock
    ) -> None:
        engine = MagicMock()
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
        engine.load_seconds = 1.25
        engine.model.cache = SimpleNamespace(
            num_bytes=2 * 1024 * 1024, capacity=16
        )
        engine.generate.return_value = iter(
            [
                GenerationEvent(
                    2,
                    0,
                    "Hello",
                    "Hello",
                    model_seconds=0.20,
                    prompt_token_count=9,
                ),
                GenerationEvent(
                    3,
                    1,
                    " world",
                    "Hello world",
                    finish_reason="max_new_tokens",
                    model_seconds=0.05,
                ),
            ]
        )
        from_model_dir.return_value = engine
        output = StringIO()
        error = StringIO()

        with patch("mini_llm.cli.time.perf_counter", side_effect=[10.0, 10.3, 10.4]):
            status = main(
                [
                    "--model",
                    "model",
                    "--prompt",
                    "raw prompt",
                    "--max-new-tokens",
                    "2",
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                    "--temperature",
                    "0.7",
                    "--top-k",
                    "20",
                    "--top-p",
                    "0.9",
                    "--seed",
                    "5",
                    "--thinking",
                ],
                output=output,
                error=error,
            )

        self.assertEqual(status, 0)
        self.assertEqual(error.getvalue(), "")
        rendered = output.getvalue()
        self.assertTrue(rendered.startswith("Hello world\n"))
        self.assertIn("prompt tokens:       9", rendered)
        self.assertIn("generated tokens:    2", rendered)
        self.assertIn("cache:               2.00 MiB (16 positions)", rendered)
        self.assertIn("time to first token: 300.00 ms", rendered)
        self.assertNotIn("prefill latency", rendered)
        self.assertIn("decode throughput:   20.00 tokens/s", rendered)
        self.assertIn("stop reason:         max_new_tokens", rendered)
        from_model_dir.assert_called_once_with(
            ANY,
            device="cpu",
            dtype="float32",
            max_seq_len=4096,
        )
        sampling = engine.generate.call_args.kwargs["sampling"]
        self.assertEqual(sampling.temperature, 0.7)
        self.assertEqual(sampling.top_k, 20)
        self.assertEqual(sampling.top_p, 0.9)
        self.assertEqual(sampling.seed, 5)
        self.assertTrue(engine.generate.call_args.kwargs["enable_thinking"])

    @patch("mini_llm.cli.Engine.from_model_dir")
    def test_no_stream_prints_final_text_once_without_metrics(
        self, from_model_dir: MagicMock
    ) -> None:
        engine = MagicMock()
        engine.generate.return_value = iter(
            [
                GenerationEvent(
                    2,
                    0,
                    "A",
                    "A",
                    model_seconds=0.1,
                    prompt_token_count=2,
                ),
                GenerationEvent(
                    3,
                    1,
                    "B",
                    "AB",
                    finish_reason="eos",
                    model_seconds=0.1,
                ),
            ]
        )
        from_model_dir.return_value = engine
        output = StringIO()

        with patch("mini_llm.cli.time.perf_counter", side_effect=[1.0, 1.1, 1.2]):
            status = main(
                [
                    "--model",
                    "model",
                    "--prompt",
                    "prompt",
                    "--no-stream",
                    "--no-metrics",
                ],
                output=output,
            )

        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "AB\n")

    def test_invalid_sampling_configuration_returns_error(self) -> None:
        error = StringIO()

        status = main(
            [
                "--model",
                "model",
                "--prompt",
                "prompt",
                "--temperature",
                "-1",
            ],
            output=StringIO(),
            error=error,
        )

        self.assertEqual(status, 2)
        self.assertIn("temperature must be", error.getvalue())


class CLIIntegrationTests(unittest.TestCase):
    def _assert_real_cpu_checkpoint_generates(self, model_dir: Path) -> None:
        output = StringIO()
        error = StringIO()

        status = main(
            [
                "--model",
                str(model_dir),
                "--prompt",
                "Say hello.",
                "--max-new-tokens",
                "1",
                "--device",
                "cpu",
                "--dtype",
                "bfloat16",
                "--no-stream",
            ],
            output=output,
            error=error,
        )

        self.assertEqual(status, 0)
        self.assertEqual(error.getvalue(), "")
        self.assertIn("metrics", output.getvalue())
        self.assertIn("generated tokens:    1", output.getvalue())
        self.assertIn("device:              cpu", output.getvalue())

    @unittest.skipUnless(
        has_local_checkpoint(QWEN_MODEL_DIR), "local Qwen3 checkpoint is unavailable"
    )
    def test_real_qwen_cpu_checkpoint_generates_text_and_metrics(self) -> None:
        self._assert_real_cpu_checkpoint_generates(QWEN_MODEL_DIR)

    @unittest.skipUnless(
        has_local_checkpoint(GRANITE_MODEL_DIR), "local Granite checkpoint is unavailable"
    )
    def test_real_granite_cpu_checkpoint_generates_text_and_metrics(self) -> None:
        self._assert_real_cpu_checkpoint_generates(GRANITE_MODEL_DIR)


if __name__ == "__main__":
    unittest.main()
