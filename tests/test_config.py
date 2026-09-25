from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from mini_llm.config import (
    ConfigError,
    GPTQQuantizationConfig,
    GraniteMoeConfig,
    Qwen3Config,
    load_config,
)


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


def valid_gptq_config() -> dict[str, object]:
    return {
        "bits": 8,
        "checkpoint_format": "gptq",
        "desc_act": False,
        "group_size": 128,
        "lm_head": False,
        "pack_dtype": "int32",
        "quant_method": "gptq",
        "sym": True,
    }


class Qwen3ConfigTests(unittest.TestCase):
    def test_parses_supported_gptq_bit_widths(self) -> None:
        for bits in (4, 8):
            with self.subTest(bits=bits):
                raw = valid_config()
                quantization = valid_gptq_config()
                quantization["bits"] = bits
                raw["quantization_config"] = quantization

                config = Qwen3Config.from_dict(raw)

                self.assertIsInstance(
                    config.quantization_config, GPTQQuantizationConfig
                )
                assert config.quantization_config is not None
                self.assertEqual(config.quantization_config.bits, bits)
                self.assertEqual(config.quantization_config.group_size, 128)

        self.assertIsNone(Qwen3Config.from_dict(valid_config()).quantization_config)

    def test_gptq_accepts_16_bit_sources_and_requires_tied_embeddings(self) -> None:
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                raw = valid_config()
                raw["torch_dtype"] = dtype
                raw["quantization_config"] = valid_gptq_config()
                Qwen3Config.from_dict(raw)

        raw["torch_dtype"] = "float32"
        with self.assertRaisesRegex(ConfigError, "float16.*bfloat16"):
            Qwen3Config.from_dict(raw)

        raw["torch_dtype"] = "float16"
        raw["tie_word_embeddings"] = False
        with self.assertRaisesRegex(ConfigError, "tied word embeddings"):
            Qwen3Config.from_dict(raw)

    def test_rejects_unsupported_gptq_metadata(self) -> None:
        for field, value in (
            ("bits", 3),
            ("checkpoint_format", "marlin"),
            ("desc_act", True),
            ("group_size", 64),
            ("lm_head", True),
            ("pack_dtype", "int16"),
            ("quant_method", "awq"),
            ("sym", False),
        ):
            with self.subTest(field=field):
                raw = valid_config()
                quantization = valid_gptq_config()
                quantization[field] = value
                raw["quantization_config"] = quantization

                with self.assertRaisesRegex(
                    ConfigError, rf"quantization_config\.{field}"
                ):
                    Qwen3Config.from_dict(raw)

    def test_rejects_wrong_gptq_metadata_types(self) -> None:
        raw = valid_config()
        raw["torch_dtype"] = "float16"
        quantization = valid_gptq_config()
        raw["quantization_config"] = quantization

        quantization["bits"] = True
        with self.assertRaisesRegex(ConfigError, "bits must be int"):
            Qwen3Config.from_dict(raw)

        quantization["bits"] = 8
        quantization["desc_act"] = 0
        with self.assertRaisesRegex(ConfigError, "desc_act must be bool"):
            Qwen3Config.from_dict(raw)

        raw["quantization_config"] = []
        with self.assertRaisesRegex(ConfigError, "must be an object"):
            Qwen3Config.from_dict(raw)

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

    def test_rejects_context_larger_than_model_limit(self) -> None:
        config = Qwen3Config.from_dict(valid_config())

        with self.assertRaisesRegex(ConfigError, "exceeds the model limit"):
            config.kv_cache_bytes(40961)

    def test_rejects_invalid_fields(self) -> None:
        for field, bad_value, regex in (
            ("num_key_value_heads", 6, "divisible"),
            ("head_dim", 127, "even for rotary embeddings"),
            ("rms_norm_eps", "small", "int or float"),
            ("bos_token_id", 151936, "within vocabulary"),
            ("model_type", "llama", "unsupported model_type"),
        ):
            with self.subTest(field=field):
                raw = valid_config()
                raw[field] = bad_value

                with self.assertRaisesRegex(ConfigError, regex):
                    Qwen3Config.from_dict(raw)


class LoadConfigTests(unittest.TestCase):
    def test_dispatches_by_model_type(self) -> None:
        unknown = valid_granite_config()
        unknown["model_type"] = "another_model"
        for name, raw, expected in (
            ("qwen3", valid_config(), Qwen3Config),
            ("granitemoe", valid_granite_config(), GraniteMoeConfig),
            ("unknown", unknown, "qwen3.*granitemoe"),
        ):
            with self.subTest(model_type=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "config.json").write_text(json.dumps(raw))

                if isinstance(expected, str):
                    with self.assertRaisesRegex(ConfigError, expected):
                        load_config(path)
                else:
                    self.assertIsInstance(load_config(path), expected)


class GraniteMoeConfigTests(unittest.TestCase):
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

    def test_rejects_invalid_granite_fields(self) -> None:
        for field, bad_value, regex in (
            ("num_experts_per_tok", 33, "cannot exceed"),
            ("tie_word_embeddings", False, "tied word embeddings"),
            ("architectures", ["AnotherModel"], "GraniteMoeForCausalLM"),
        ):
            with self.subTest(field=field):
                raw = valid_granite_config()
                raw[field] = bad_value

                with self.assertRaisesRegex(ConfigError, regex):
                    GraniteMoeConfig.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
