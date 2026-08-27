from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file
import torch

from mini_llm.checkpoint import SafeTensorCheckpoint
from mini_llm.config import Qwen3Config
from mini_llm.model import Qwen3DecoderLayer, Qwen3ForCausalLM
from mini_llm.nn import RotaryEmbedding, build_position_ids


def _tiny_config_data(*, num_hidden_layers: int = 2) -> dict[str, object]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _tiny_config(*, num_hidden_layers: int = 2) -> Qwen3Config:
    return Qwen3Config.from_dict(
        _tiny_config_data(num_hidden_layers=num_hidden_layers)
    )


class Qwen3DecoderLayerTests(unittest.TestCase):
    def test_matches_explicit_pre_norm_residual_composition(self) -> None:
        torch.manual_seed(19)
        config = _tiny_config(num_hidden_layers=1)
        layer = Qwen3DecoderLayer(config).eval()
        inputs = torch.randn(1, 3, config.hidden_size)
        rope = RotaryEmbedding(config.head_dim, theta=config.rope_theta)
        cosine, sine = rope(build_position_ids(3))

        attention_output = layer.self_attn(
            layer.input_layernorm(inputs), cosine, sine
        )
        after_attention = inputs + attention_output
        expected = after_attention + layer.mlp(
            layer.post_attention_layernorm(after_attention)
        )

        actual = layer(inputs, cosine, sine)

        torch.testing.assert_close(actual, expected)


class Qwen3ForCausalLMTests(unittest.TestCase):
    def test_forward_returns_finite_logits_for_every_token(self) -> None:
        config = _tiny_config()
        model = Qwen3ForCausalLM(config).eval()

        logits = model(torch.tensor([[1, 4, 7], [2, 8, 9]]))

        self.assertEqual(logits.shape, (2, 3, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_automatic_positions_match_explicit_positions(self) -> None:
        config = _tiny_config()
        model = Qwen3ForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 4, 7]])
        positions = torch.tensor([[0, 1, 2]])

        automatic = model(input_ids)
        explicit = model(input_ids, position_ids=positions)

        torch.testing.assert_close(automatic, explicit)

    def test_module_names_match_checkpoint_contract(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config())
        names = set(model.state_dict())

        self.assertIn("model.embed_tokens.weight", names)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("model.layers.0.self_attn.q_norm.weight", names)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", names)
        self.assertIn("model.layers.0.input_layernorm.weight", names)
        self.assertIn("model.norm.weight", names)
        self.assertIn("lm_head.weight", names)
        self.assertNotIn("model.rotary_emb.inverse_frequencies", names)

    def test_meta_model_assigns_checkpoint_tensors_without_placeholders(self) -> None:
        config = _tiny_config(num_hidden_layers=1)
        source = Qwen3ForCausalLM(config).eval()
        tensors = {name: value.detach() for name, value in source.state_dict().items()}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.safetensors"
            save_file(tensors, checkpoint_path)
            checkpoint = SafeTensorCheckpoint(checkpoint_path)
            with torch.device("meta"):
                loaded = Qwen3ForCausalLM(config)
            loaded.load_checkpoint(checkpoint)

        self.assertEqual(loaded.lm_head.weight.device.type, "cpu")
        self.assertEqual(loaded.model.rotary_emb.inverse_frequencies.device.type, "cpu")
        torch.testing.assert_close(loaded.lm_head.weight, source.lm_head.weight)

    def test_from_model_dir_validates_and_loads_complete_checkpoint(self) -> None:
        config_data = _tiny_config_data(num_hidden_layers=1)
        source = Qwen3ForCausalLM(Qwen3Config.from_dict(config_data)).eval()
        tensors = {name: value.detach() for name, value in source.state_dict().items()}
        input_ids = torch.tensor([[1, 4, 7]])

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps(config_data))
            save_file(tensors, model_dir / "model.safetensors")

            loaded = Qwen3ForCausalLM.from_model_dir(model_dir)
            actual = loaded(input_ids)

        expected = source(input_ids)
        self.assertFalse(loaded.training)
        self.assertFalse(any(parameter.is_meta for parameter in loaded.parameters()))
        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_token_ids_and_position_shape(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config())

        with self.assertRaisesRegex(ValueError, "within vocabulary"):
            model(torch.tensor([[32]]))
        with self.assertRaisesRegex(ValueError, "same.*shape"):
            model(torch.tensor([[1, 2]]), position_ids=torch.tensor([[0]]))


if __name__ == "__main__":
    unittest.main()
