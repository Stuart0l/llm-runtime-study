from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import torch

from benchmarks.__main__ import BenchmarkError, run
from benchmarks.common import PromptCase


def _args(**overrides: object) -> Namespace:
    values = {
        "model": [Path("qwen-gptq")],
        "benchmark": ["end-to-end"],
        "device": None,
        "prompt_lengths": [32],
        "warmups": 0,
        "repeats": 1,
        "decode_tokens": 1,
    }
    values.update(overrides)
    return Namespace(**values)


class QuantizedBenchmarkTests(unittest.TestCase):
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
            run(_args(), output=StringIO())

        from_model_dir.assert_called_once_with(
            Path("qwen-gptq"), device="cuda", dtype="float16", max_seq_len=97
        )
        engine.to.assert_not_called()

    @patch("benchmarks.__main__.Engine.from_model_dir")
    @patch("benchmarks.__main__.load_config")
    def test_rejects_explicit_cpu_before_loading_gptq(
        self, load_config: MagicMock, from_model_dir: MagicMock
    ) -> None:
        load_config.return_value = SimpleNamespace(quantization_config=object())

        with self.assertRaisesRegex(BenchmarkError, "CUDA only"):
            run(_args(device=["cpu"]), output=StringIO())

        from_model_dir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
