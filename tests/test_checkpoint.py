from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from mini_llm.checkpoint import (
    CheckpointError,
    CheckpointValidationError,
    SafeTensorCheckpoint,
    expected_qwen3_tensors,
    validate_qwen3_checkpoint,
)
from mini_llm.config import Qwen3Config


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"


def tiny_config() -> Qwen3Config:
    return Qwen3Config.from_dict(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "bos_token_id": 1,
            "eos_token_id": 2,
        }
    )


def tensors_for(config: Qwen3Config) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(spec.shape, dtype=torch.bfloat16)
        for name, spec in expected_qwen3_tensors(config).items()
    }


class CheckpointSchemaTests(unittest.TestCase):
    def test_expected_local_model_schema_has_311_tensors(self) -> None:
        config = Qwen3Config.from_model_dir(MODEL_DIR)

        specs = expected_qwen3_tensors(config)

        self.assertEqual(len(specs), 311)
        self.assertEqual(
            specs["model.layers.0.self_attn.q_proj.weight"].shape,
            (2048, 1024),
        )
        self.assertEqual(
            specs["model.layers.0.self_attn.k_proj.weight"].shape,
            (1024, 1024),
        )

    def test_attention_bias_configuration_adds_qkv_bias_tensors(self) -> None:
        raw = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000,
            "hidden_act": "silu",
            "attention_bias": True,
            "attention_dropout": 0.0,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "bos_token_id": 1,
            "eos_token_id": 2,
        }

        specs = expected_qwen3_tensors(Qwen3Config.from_dict(raw))

        self.assertEqual(specs["model.layers.0.self_attn.q_proj.bias"].shape, (8,))
        self.assertEqual(specs["model.layers.0.self_attn.k_proj.bias"].shape, (4,))
        self.assertEqual(specs["model.layers.0.self_attn.v_proj.bias"].shape, (4,))

    def test_validates_complete_synthetic_checkpoint(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors_for(config), path, metadata={"format": "pt"})
            checkpoint = SafeTensorCheckpoint(path)

            validate_qwen3_checkpoint(checkpoint, config)

        self.assertEqual(checkpoint.tensor_count, 14)
        self.assertEqual(checkpoint.metadata, {"format": "pt"})

    def test_reports_missing_and_unexpected_tensors_together(self) -> None:
        config = tiny_config()
        tensors = tensors_for(config)
        del tensors["model.norm.weight"]
        tensors["not.a.model.tensor"] = torch.zeros(1, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, path)
            checkpoint = SafeTensorCheckpoint(path)

            with self.assertRaises(CheckpointValidationError) as caught:
                validate_qwen3_checkpoint(checkpoint, config)

        message = str(caught.exception)
        self.assertIn("missing tensors: model.norm.weight", message)
        self.assertIn("unexpected tensors: not.a.model.tensor", message)

    def test_reports_shape_and_dtype_mismatches_together(self) -> None:
        config = tiny_config()
        tensors = tensors_for(config)
        tensors["model.norm.weight"] = torch.zeros(7, dtype=torch.bfloat16)
        tensors["lm_head.weight"] = torch.zeros((16, 8), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, path)
            checkpoint = SafeTensorCheckpoint(path)

            with self.assertRaises(CheckpointValidationError) as caught:
                validate_qwen3_checkpoint(checkpoint, config)

        message = str(caught.exception)
        self.assertIn("shape mismatch for model.norm.weight", message)
        self.assertIn("dtype mismatch for lm_head.weight", message)

    def test_get_tensor_materializes_only_requested_value(self) -> None:
        config = tiny_config()
        tensors = tensors_for(config)
        tensors["model.norm.weight"] = torch.arange(8, dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, path)
            checkpoint = SafeTensorCheckpoint(path)

            tensor = checkpoint.get_tensor("model.norm.weight")

        self.assertEqual(tuple(tensor.shape), (8,))
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tensor.tolist(), list(range(8)))

    def test_get_tensor_rejects_unknown_name(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors_for(config), path)
            checkpoint = SafeTensorCheckpoint(path)

            with self.assertRaisesRegex(CheckpointError, "not present"):
                checkpoint.get_tensor("missing.weight")


@unittest.skipUnless(MODEL_DIR.is_dir(), "local Qwen3 checkpoint is unavailable")
class LocalCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Qwen3Config.from_model_dir(MODEL_DIR)
        cls.checkpoint = SafeTensorCheckpoint.from_model_dir(MODEL_DIR)

    def test_local_checkpoint_matches_complete_schema(self) -> None:
        validate_qwen3_checkpoint(self.checkpoint, self.config)
        self.assertEqual(self.checkpoint.tensor_count, 311)

    def test_local_manifest_does_not_materialize_payloads(self) -> None:
        info = self.checkpoint.tensor_info(
            "model.layers.0.self_attn.q_norm.weight"
        )

        self.assertEqual(info.shape, (128,))
        self.assertEqual(info.dtype, "BF16")
        self.assertEqual(info.num_bytes, 256)

    def test_loads_one_local_tensor_on_cpu(self) -> None:
        tensor = self.checkpoint.get_tensor(
            "model.layers.0.self_attn.q_norm.weight"
        )

        self.assertEqual(tuple(tensor.shape), (128,))
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tensor.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
