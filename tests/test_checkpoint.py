from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from tests.reference_support import has_local_checkpoint

from mini_llm.checkpoint import (
    CheckpointError,
    CheckpointValidationError,
    SafeTensorCheckpoint,
    expected_granite_moe_tensors,
    expected_qwen3_tensors,
    validate_checkpoint,
    validate_granite_moe_checkpoint,
    validate_qwen3_checkpoint,
)
from mini_llm.config import GraniteMoeConfig, Qwen3Config


QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


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


def tiny_granite_config() -> GraniteMoeConfig:
    return GraniteMoeConfig.from_dict(
        {
            "architectures": ["GraniteMoeForCausalLM"],
            "model_type": "granitemoe",
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_multiplier": 0.5,
            "embedding_multiplier": 2.0,
            "residual_multiplier": 0.25,
            "logits_scaling": 4.0,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
        }
    )


def granite_tensors_for(config: GraniteMoeConfig) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(spec.shape, dtype=torch.bfloat16)
        for name, spec in expected_granite_moe_tensors(config).items()
    }


def save_sharded_checkpoint(
    model_dir: Path,
    tensors: dict[str, torch.Tensor],
    *,
    metadata: dict[str, str] | None = None,
) -> tuple[Path, Path, dict[str, str]]:
    """Split tensors across two shards and write a Hugging Face-style index."""

    names = sorted(tensors)
    split = len(names) // 2
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    first_names = names[:split]
    second_names = names[split:]
    save_file(
        {name: tensors[name] for name in first_names},
        model_dir / first_name,
        metadata=metadata,
    )
    save_file(
        {name: tensors[name] for name in second_names},
        model_dir / second_name,
        metadata=metadata,
    )
    weight_map = {
        name: first_name if name in first_names else second_name for name in names
    }
    total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        ),
        encoding="utf-8",
    )
    return model_dir / first_name, model_dir / second_name, weight_map


class CheckpointSchemaTests(unittest.TestCase):
    def test_expected_local_model_schema_has_311_tensors(self) -> None:
        config = Qwen3Config.from_model_dir(QWEN_MODEL_DIR)

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


class GraniteCheckpointSchemaTests(unittest.TestCase):
    def test_describes_packed_experts_and_tied_embeddings(self) -> None:
        config = tiny_granite_config()

        specs = expected_granite_moe_tensors(config)

        self.assertEqual(len(specs), 11)
        self.assertNotIn("lm_head.weight", specs)
        self.assertEqual(
            specs[
                "model.layers.0.block_sparse_moe.router.layer.weight"
            ].shape,
            (4, 8),
        )
        self.assertEqual(
            specs[
                "model.layers.0.block_sparse_moe.input_linear.weight"
            ].shape,
            (4, 8, 8),
        )
        self.assertEqual(
            specs[
                "model.layers.0.block_sparse_moe.output_linear.weight"
            ].shape,
            (4, 8, 4),
        )

    def test_validates_complete_synthetic_granite_checkpoint(self) -> None:
        config = tiny_granite_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(granite_tensors_for(config), path)
            checkpoint = SafeTensorCheckpoint(path)

            validate_granite_moe_checkpoint(checkpoint, config)
            validate_checkpoint(checkpoint, config)

        self.assertEqual(checkpoint.tensor_count, 11)

    def test_reports_packed_expert_shape_and_dtype_errors(self) -> None:
        config = tiny_granite_config()
        tensors = granite_tensors_for(config)
        input_name = "model.layers.0.block_sparse_moe.input_linear.weight"
        output_name = "model.layers.0.block_sparse_moe.output_linear.weight"
        tensors[input_name] = torch.zeros((4, 4, 8), dtype=torch.bfloat16)
        tensors[output_name] = torch.zeros((4, 8, 4), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, path)
            checkpoint = SafeTensorCheckpoint(path)

            with self.assertRaises(CheckpointValidationError) as caught:
                validate_granite_moe_checkpoint(checkpoint, config)

        message = str(caught.exception)
        self.assertIn(f"shape mismatch for {input_name}", message)
        self.assertIn(f"dtype mismatch for {output_name}", message)

    def test_rejects_separate_lm_head_for_tied_embeddings(self) -> None:
        config = tiny_granite_config()
        tensors = granite_tensors_for(config)
        tensors["lm_head.weight"] = torch.zeros((16, 8), dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(tensors, path)
            checkpoint = SafeTensorCheckpoint(path)

            with self.assertRaisesRegex(
                CheckpointValidationError, "unexpected tensors: lm_head.weight"
            ):
                validate_granite_moe_checkpoint(checkpoint, config)


class ShardedCheckpointTests(unittest.TestCase):
    def test_combines_manifests_and_loads_tensors_from_each_shard(self) -> None:
        config = tiny_config()
        tensors = tensors_for(config)
        tensors["model.embed_tokens.weight"] = torch.arange(
            128, dtype=torch.bfloat16
        ).reshape(16, 8)
        tensors["model.norm.weight"] = torch.arange(8, dtype=torch.bfloat16)

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            first, second, weight_map = save_sharded_checkpoint(
                model_dir, tensors, metadata={"format": "pt"}
            )
            checkpoint = SafeTensorCheckpoint.from_model_dir(model_dir)
            validate_qwen3_checkpoint(checkpoint, config)

            names = ["model.embed_tokens.weight", "model.norm.weight"]
            loaded = checkpoint.get_tensors(names)

        self.assertTrue(checkpoint.is_sharded)
        self.assertEqual(checkpoint.shard_paths, (first, second))
        self.assertEqual(checkpoint.metadata, {"format": "pt"})
        self.assertEqual(
            checkpoint.index_metadata,
            {
                "total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors.values()
                )
            },
        )
        self.assertNotEqual(weight_map[names[0]], weight_map[names[1]])
        torch.testing.assert_close(loaded[names[0]], tensors[names[0]])
        torch.testing.assert_close(loaded[names[1]], tensors[names[1]])

    def test_rejects_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            index = {
                "metadata": {"total_size": 16},
                "weight_map": {"weight": "missing.safetensors"},
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            with self.assertRaisesRegex(CheckpointError, "shard does not exist"):
                SafeTensorCheckpoint.from_model_dir(model_dir)

@unittest.skipUnless(
    has_local_checkpoint(QWEN_MODEL_DIR), "local Qwen3 checkpoint is unavailable"
)
class LocalCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Qwen3Config.from_model_dir(QWEN_MODEL_DIR)
        cls.checkpoint = SafeTensorCheckpoint.from_model_dir(QWEN_MODEL_DIR)

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


@unittest.skipUnless(
    has_local_checkpoint(GRANITE_MODEL_DIR), "local Granite checkpoint is unavailable"
)
class GraniteLocalCheckpointIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GraniteMoeConfig.from_model_dir(GRANITE_MODEL_DIR)
        cls.checkpoint = SafeTensorCheckpoint.from_model_dir(GRANITE_MODEL_DIR)

    def test_local_checkpoint_matches_all_218_tensor_specs(self) -> None:
        validate_granite_moe_checkpoint(self.checkpoint, self.config)

        self.assertEqual(self.checkpoint.tensor_count, 218)
        self.assertEqual(
            sum(tensor.num_elements for tensor in self.checkpoint.manifest),
            self.config.total_parameter_estimate,
        )

    def test_local_packed_expert_headers_match_configuration(self) -> None:
        input_info = self.checkpoint.tensor_info(
            "model.layers.0.block_sparse_moe.input_linear.weight"
        )
        output_info = self.checkpoint.tensor_info(
            "model.layers.0.block_sparse_moe.output_linear.weight"
        )
        router_info = self.checkpoint.tensor_info(
            "model.layers.0.block_sparse_moe.router.layer.weight"
        )

        self.assertEqual(input_info.shape, (32, 1024, 1024))
        self.assertEqual(output_info.shape, (32, 1024, 512))
        self.assertEqual(router_info.shape, (32, 1024))
        self.assertEqual(input_info.dtype, "BF16")

    def test_tied_embedding_checkpoint_has_no_lm_head(self) -> None:
        names = {tensor.name for tensor in self.checkpoint.manifest}

        self.assertIn("model.embed_tokens.weight", names)
        self.assertNotIn("lm_head.weight", names)


if __name__ == "__main__":
    unittest.main()
