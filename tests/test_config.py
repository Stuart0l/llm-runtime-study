from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mini_llm.config import ConfigError, GraniteMoeConfig, Qwen3Config, load_config
from examples.inspect_config import render_summary


def valid_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }


def valid_granite_config() -> dict[str, object]:
    return {
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
        "vocab_size": 49155,
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_500_000,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_multiplier": 0.015625,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "logits_scaling": 6.0,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }


class Qwen3ConfigTests(unittest.TestCase):
    def test_shared_loader_dispatches_qwen_by_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(valid_config()))

            config = load_config(path)

        self.assertIsInstance(config, Qwen3Config)

    def test_loads_from_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(valid_config()))

            config = Qwen3Config.from_model_dir(path)

        self.assertEqual(config.model_type, "qwen3")
        self.assertEqual(config.architectures, ("Qwen3ForCausalLM",))

    def test_derives_attention_dimensions_without_assuming_hidden_size_per_head(self) -> None:
        config = Qwen3Config.from_dict(valid_config())

        self.assertEqual(config.query_projection_size, 2048)
        self.assertEqual(config.kv_projection_size, 1024)
        self.assertEqual(config.queries_per_kv_head, 2)

    def test_estimates_dense_kv_cache_size(self) -> None:
        config = Qwen3Config.from_dict(valid_config())

        size = config.kv_cache_bytes(4096, dtype="float16")

        self.assertEqual(size, 28 * 2 * 1 * 8 * 4096 * 128 * 2)
        self.assertEqual(size, 469_762_048)

    def test_accepts_multiple_eos_tokens(self) -> None:
        raw = valid_config()
        raw["eos_token_id"] = [151645, 151643]

        config = Qwen3Config.from_dict(raw)

        self.assertEqual(config.eos_token_ids, (151645, 151643))

    def test_rejects_incompatible_gqa_heads(self) -> None:
        raw = valid_config()
        raw["num_key_value_heads"] = 6

        with self.assertRaisesRegex(ConfigError, "divisible"):
            Qwen3Config.from_dict(raw)

    def test_rejects_odd_head_dimension(self) -> None:
        raw = valid_config()
        raw["head_dim"] = 127

        with self.assertRaisesRegex(ConfigError, "even for rotary embeddings"):
            Qwen3Config.from_dict(raw)

    def test_reports_invalid_numeric_field_as_config_error(self) -> None:
        raw = valid_config()
        raw["rms_norm_eps"] = "small"

        with self.assertRaisesRegex(ConfigError, "int or float"):
            Qwen3Config.from_dict(raw)

    def test_rejects_context_larger_than_model_limit(self) -> None:
        config = Qwen3Config.from_dict(valid_config())

        with self.assertRaisesRegex(ConfigError, "exceeds the model limit"):
            config.kv_cache_bytes(40961)

    def test_rejects_token_id_outside_vocabulary(self) -> None:
        raw = valid_config()
        raw["bos_token_id"] = 151936

        with self.assertRaisesRegex(ConfigError, "within vocabulary"):
            Qwen3Config.from_dict(raw)

    def test_rejects_another_model_family(self) -> None:
        raw = valid_config()
        raw["model_type"] = "llama"

        with self.assertRaisesRegex(ConfigError, "unsupported model_type"):
            Qwen3Config.from_dict(raw)

    def test_summary_exposes_architecture_and_cache_numbers(self) -> None:
        config = Qwen3Config.from_dict(valid_config())

        summary = render_summary(
            config,
            model_dir=Path("model"),
            max_seq_len=4096,
            cache_dtype="float16",
            batch_size=1,
        )

        self.assertIn("query projection:      2,048", summary)
        self.assertIn("total:                 448.00 MiB", summary)


class GraniteMoeConfigTests(unittest.TestCase):
    def test_load_config_dispatches_by_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(valid_granite_config()))

            config = load_config(path)

        self.assertIsInstance(config, GraniteMoeConfig)
        self.assertEqual(config.architectures, ("GraniteMoeForCausalLM",))

    def test_derives_attention_expert_and_cache_dimensions(self) -> None:
        config = GraniteMoeConfig.from_dict(valid_granite_config())

        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.query_projection_size, 1024)
        self.assertEqual(config.kv_projection_size, 512)
        self.assertEqual(config.queries_per_kv_head, 2)
        self.assertEqual(config.parameters_per_expert, 1_572_864)
        self.assertEqual(config.total_expert_parameters, 1_207_959_552)
        self.assertEqual(config.active_expert_parameters, 301_989_888)
        self.assertEqual(config.total_parameter_estimate, 1_334_628_352)
        self.assertEqual(config.active_parameter_estimate, 428_658_688)
        self.assertEqual(config.kv_cache_bytes(4096), 201_326_592)

        summary = render_summary(
            config,
            model_dir=Path("granite"),
            max_seq_len=4096,
            cache_dtype="float16",
            batch_size=1,
        )
        self.assertIn("Granite MoE configuration: valid", summary)
        self.assertIn("total / active:        32 / 8 per token", summary)
        self.assertIn("active parameters:     428,658,688", summary)
        self.assertIn("total:                 192.00 MiB", summary)

    def test_rejects_invalid_expert_and_attention_ratios(self) -> None:
        too_many_active = valid_granite_config()
        too_many_active["num_experts_per_tok"] = 33
        with self.assertRaisesRegex(ConfigError, "cannot exceed"):
            GraniteMoeConfig.from_dict(too_many_active)

        invalid_heads = valid_granite_config()
        invalid_heads["num_key_value_heads"] = 6
        with self.assertRaisesRegex(ConfigError, "divisible"):
            GraniteMoeConfig.from_dict(invalid_heads)

    def test_rejects_untied_embeddings_and_non_granite_architecture(self) -> None:
        untied = valid_granite_config()
        untied["tie_word_embeddings"] = False
        with self.assertRaisesRegex(ConfigError, "tied word embeddings"):
            GraniteMoeConfig.from_dict(untied)

        wrong_architecture = valid_granite_config()
        wrong_architecture["architectures"] = ["AnotherModel"]
        with self.assertRaisesRegex(ConfigError, "GraniteMoeForCausalLM"):
            GraniteMoeConfig.from_dict(wrong_architecture)

    def test_rejects_unknown_family_at_dispatch_boundary(self) -> None:
        raw = valid_granite_config()
        raw["model_type"] = "another_model"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(raw))

            with self.assertRaisesRegex(ConfigError, "qwen3.*granitemoe"):
                load_config(path)


GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


@unittest.skipUnless(
    GRANITE_MODEL_DIR.is_dir(), "local Granite 3.1 checkpoint is unavailable"
)
class GraniteMoeConfigIntegrationTests(unittest.TestCase):
    def test_local_config_matches_supported_architecture(self) -> None:
        config = load_config(GRANITE_MODEL_DIR)

        self.assertIsInstance(config, GraniteMoeConfig)
        self.assertEqual(config.vocab_size, 49155)
        self.assertEqual(config.total_parameter_estimate, 1_334_628_352)


if __name__ == "__main__":
    unittest.main()
