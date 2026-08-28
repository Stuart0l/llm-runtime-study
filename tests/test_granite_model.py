from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from mini_llm.config import GraniteMoeConfig
from mini_llm.granite_model import (
    GraniteMoeDecoderLayer,
    GraniteMoeForCausalLM,
    GraniteMoeModel,
)
from mini_llm.nn import RotaryEmbedding, build_position_ids
from mini_llm.tokenizer import GraniteTokenizer
from tests.reference_support import (
    HAS_TRANSFORMERS,
    formatted_input,
    run_mini_runtime,
    run_transformers,
)


_GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


def _tiny_config_data(*, num_hidden_layers: int = 2) -> dict[str, object]:
    return {
        "architectures": ["GraniteMoeForCausalLM"],
        "model_type": "granitemoe",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 4,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "attention_multiplier": 0.5,
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "logits_scaling": 6.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
    }


def _tiny_config(*, num_hidden_layers: int = 2) -> GraniteMoeConfig:
    return GraniteMoeConfig.from_dict(
        _tiny_config_data(num_hidden_layers=num_hidden_layers)
    )


class GraniteMoeDecoderLayerTests(unittest.TestCase):
    def test_matches_explicit_scaled_residual_composition(self) -> None:
        torch.manual_seed(43)
        config = _tiny_config(num_hidden_layers=1)
        layer = GraniteMoeDecoderLayer(config).eval()
        inputs = torch.randn(1, 3, config.hidden_size)
        rope = RotaryEmbedding(config.head_dim, theta=config.rope_theta)
        cosine, sine = rope(build_position_ids(3))

        attention_output = layer.self_attn(
            layer.input_layernorm(inputs), cosine, sine
        )
        after_attention = inputs + attention_output * config.residual_multiplier
        moe_output, _ = layer.block_sparse_moe(
            layer.post_attention_layernorm(after_attention)
        )
        expected = after_attention + moe_output * config.residual_multiplier

        actual = layer(inputs, cosine, sine)

        torch.testing.assert_close(actual, expected)


class GraniteMoeForCausalLMTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_TRANSFORMERS,
        "the optional Transformers reference dependency is not installed",
    )
    @unittest.skipUnless(
        (_GRANITE_MODEL_DIR / "model.safetensors").is_file(),
        "the local Granite 3.1 checkpoint is not available",
    )
    def test_formatted_chat_generation_matches_transformers_sdpa(self) -> None:
        """Compare relevant next-token logits and cached greedy decisions.

        Packed BF16 expert dispatch is not bit-identical to Transformers because
        grouping and accumulation can round differently. Require closely aligned
        logit vectors and identical greedy tokens instead.
        """

        tokenizer = GraniteTokenizer.from_model_dir(_GRANITE_MODEL_DIR)
        input_ids = formatted_input(
            tokenizer,
            "Define sparse routing briefly.",
        )

        model = GraniteMoeForCausalLM.from_model_dir(_GRANITE_MODEL_DIR)
        actual = run_mini_runtime(model, input_ids, generated_tokens=4)
        del model
        gc.collect()
        expected = run_transformers(
            _GRANITE_MODEL_DIR,
            input_ids,
            generated_tokens=4,
        )

        torch.testing.assert_close(
            actual.full_logits,
            actual.prefill_logits,
            rtol=0.0,
            atol=0.0,
        )
        relevant_pairs = [
            (actual.prefill_logits[:, -1], expected.prefill_logits[:, -1]),
            *zip(actual.decode_logits, expected.decode_logits, strict=True),
        ]
        for ours, reference in relevant_pairs:
            similarity = F.cosine_similarity(
                ours.reshape(1, -1), reference.reshape(1, -1)
            )
            self.assertGreater(float(similarity), 0.99)
            self.assertLess(float((ours - reference).abs().max()), 2.0)
            self.assertTrue(
                torch.equal(ours.argmax(dim=-1), reference.argmax(dim=-1))
            )
        self.assertEqual(actual.token_ids, expected.token_ids)

        our_text = tokenizer.decode(actual.token_ids)
        reference_text = tokenizer.decode(expected.token_ids)
        self.assertEqual(our_text, reference_text)
        print("\nGranite formatted-chat reference comparison")
        print(f"  Our output: {our_text!r}")
        print(f"  HF output:  {reference_text!r}")

    def test_prefill_and_decode_match_uncached_logits(self) -> None:
        torch.manual_seed(59)
        model = GraniteMoeForCausalLM(_tiny_config()).eval()
        input_ids = torch.tensor([[1, 4, 7, 9]])
        reference = model(input_ids)
        model.setup_cache(capacity=input_ids.shape[1])

        prefill = model.prefill(input_ids[:, :2])
        first_decode = model.decode(input_ids[:, 2:3])
        second_decode = model.decode(input_ids[:, 3:4])

        torch.testing.assert_close(prefill, reference[:, :2])
        torch.testing.assert_close(first_decode, reference[:, 2:3])
        torch.testing.assert_close(second_decode, reference[:, 3:4])
        assert model.cache is not None
        self.assertEqual(model.cache.length, 4)

    def test_model_applies_embedding_multiplier_before_decoder(self) -> None:
        torch.manual_seed(47)
        config = _tiny_config(num_hidden_layers=1)
        model = GraniteMoeModel(config).eval()
        input_ids = torch.tensor([[1, 4, 7]])
        position_ids = build_position_ids(3)

        hidden_states = model.embed_tokens(input_ids) * config.embedding_multiplier
        cosine, sine = model.rotary_emb(position_ids)
        for layer in model.layers:
            hidden_states = layer(hidden_states, cosine, sine)
        expected = model.norm(hidden_states)

        actual = model(input_ids, position_ids=position_ids)

        torch.testing.assert_close(actual, expected)

    def test_tied_embedding_projection_and_logits_scaling(self) -> None:
        torch.manual_seed(53)
        config = _tiny_config(num_hidden_layers=1)
        model = GraniteMoeForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 4, 7]])

        hidden_states = model.model(input_ids)
        expected = F.linear(hidden_states, model.model.embed_tokens.weight)
        expected = expected / config.logits_scaling

        actual = model(input_ids)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(hasattr(model, "lm_head"))

    def test_forward_returns_finite_logits_for_every_token(self) -> None:
        model = GraniteMoeForCausalLM(_tiny_config()).eval()

        logits = model(torch.tensor([[1, 4, 7], [2, 8, 9]]))

        self.assertEqual(logits.shape, (2, 3, model.config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_module_names_match_218_tensor_checkpoint_contract(self) -> None:
        config = GraniteMoeConfig.from_model_dir(_GRANITE_MODEL_DIR)

        with torch.device("meta"):
            model = GraniteMoeForCausalLM(config)
        names = set(model.state_dict())

        self.assertEqual(len(model.model.layers), 24)
        self.assertEqual(len(names), 218)
        self.assertIn("model.embed_tokens.weight", names)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn(
            "model.layers.0.block_sparse_moe.router.layer.weight", names
        )
        self.assertIn(
            "model.layers.0.block_sparse_moe.input_linear.weight", names
        )
        self.assertIn("model.norm.weight", names)
        self.assertNotIn("lm_head.weight", names)
        self.assertNotIn("model.rotary_emb.inverse_frequencies", names)

    def test_meta_model_assigns_complete_checkpoint_and_enters_eval_mode(self) -> None:
        config_data = _tiny_config_data(num_hidden_layers=1)
        config = GraniteMoeConfig.from_dict(config_data)
        source = GraniteMoeForCausalLM(config).eval()
        tensors = {
            name: value.detach() for name, value in source.state_dict().items()
        }
        input_ids = torch.tensor([[1, 4, 7]])

        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps(config_data))
            save_file(tensors, model_dir / "model.safetensors")
            loaded = GraniteMoeForCausalLM.from_model_dir(model_dir)
            actual = loaded(input_ids)

        expected = source(input_ids)
        self.assertFalse(loaded.training)
        self.assertFalse(any(parameter.is_meta for parameter in loaded.parameters()))
        self.assertEqual(loaded.input_device.type, "cpu")
        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_token_ids_and_position_shape(self) -> None:
        model = GraniteMoeForCausalLM(_tiny_config())

        with self.assertRaisesRegex(ValueError, "within vocabulary"):
            model(torch.tensor([[32]]))
        with self.assertRaisesRegex(ValueError, "same.*shape"):
            model(torch.tensor([[1, 2]]), position_ids=torch.tensor([[0]]))


if __name__ == "__main__":
    unittest.main()
