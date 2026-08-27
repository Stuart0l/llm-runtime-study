from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mini_llm.config import ConfigError, Qwen3Config
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


class Qwen3ConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
