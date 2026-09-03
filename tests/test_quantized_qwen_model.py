from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from mini_llm.checkpoint import expected_qwen3_tensors
from mini_llm.config import Qwen3Config
from mini_llm.quantization import GPTQMarlinLinear
from mini_llm.qwen_model import Qwen3ForCausalLM


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b-int8"


def _config_data() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 16,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "quantization_config": {
            "bits": 8,
            "checkpoint_format": "gptq",
            "desc_act": False,
            "group_size": 128,
            "lm_head": False,
            "pack_dtype": "int32",
            "quant_method": "gptq",
            "sym": True,
        },
    }


def _checkpoint_tensors(config: Qwen3Config) -> dict[str, torch.Tensor]:
    tensors = {
        name: torch.zeros(
            spec.shape,
            dtype={"F16": torch.float16, "I32": torch.int32}[spec.dtype],
        )
        for name, spec in expected_qwen3_tensors(config).items()
    }
    group_size = config.quantization_config.group_size
    for name, tensor in tensors.items():
        if name.endswith(".g_idx"):
            tensor.copy_(
                torch.arange(tensor.numel(), dtype=torch.int32).div(
                    group_size, rounding_mode="floor"
                )
            )
    return tensors


class QuantizedQwenModelTests(unittest.TestCase):
    def test_module_hierarchy_exactly_matches_gptq_checkpoint(self) -> None:
        config = Qwen3Config.from_dict(_config_data())
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)

        projections = [
            module
            for module in model.modules()
            if isinstance(module, GPTQMarlinLinear)
        ]
        self.assertEqual(len(projections), 7)
        self.assertIsNone(model.lm_head)
        self.assertEqual(
            set(model.state_dict()), set(expected_qwen3_tensors(config))
        )

    def test_strictly_loads_synthetic_gptq_checkpoint_on_cpu(self) -> None:
        config_data = _config_data()
        config = Qwen3Config.from_dict(config_data)

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps(config_data))
            save_file(_checkpoint_tensors(config), model_dir / "model.safetensors")

            model = Qwen3ForCausalLM.from_model_dir(model_dir)

        projections = [
            module
            for module in model.modules()
            if isinstance(module, GPTQMarlinLinear)
        ]
        self.assertFalse(model.training)
        self.assertEqual(len(projections), 7)
        self.assertFalse(any(parameter.is_meta for parameter in model.parameters()))
        self.assertTrue(
            all(projection.qweight.device.type == "cpu" for projection in projections)
        )
        self.assertTrue(
            all(projection.workspace.numel() == 0 for projection in projections)
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and MODEL_DIR.is_dir(),
        "requires CUDA and the real GPTQ checkpoint",
    )
    def test_real_checkpoint_runs_end_to_end_with_fused_marlin(self) -> None:
        model = Qwen3ForCausalLM.from_model_dir(MODEL_DIR)
        try:
            projections = [
                module
                for module in model.modules()
                if isinstance(module, GPTQMarlinLinear)
            ]
            self.assertEqual(len(projections), 196)

            model.to(device="cuda", dtype=torch.float16)
            self.assertTrue(
                all(
                    projection.qweight.device.type == "cpu"
                    for projection in projections
                )
            )
            model.prepare_quantized("cuda")
            model.materialize_derived_buffers("cuda")

            logits = model(torch.tensor([[1]], device="cuda"))

            self.assertEqual(logits.shape, (1, 1, model.config.vocab_size))
            self.assertEqual(logits.dtype, torch.float16)
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(
                all(
                    projection.qweight.device.type == "cuda"
                    for projection in projections
                )
            )
            self.assertTrue(
                all(
                    projection._canonical_qweight is not None
                    and projection._canonical_qweight.device.type == "cpu"
                    for projection in projections
                )
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
