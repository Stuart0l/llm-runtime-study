from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file
import torch

from mini_llm.checkpoint import expected_qwen3_tensors
from mini_llm.config import Qwen3Config
from mini_llm.model.qwen import Qwen3DecoderLayer, Qwen3ForCausalLM
from mini_llm.nn import RotaryEmbedding, build_position_ids
from mini_llm.tokenizer import Qwen3Tokenizer
from tests.reference_support import (
    HAS_TRANSFORMERS,
    formatted_input,
    run_mini_runtime,
    run_transformers,
)


_QWEN_MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"


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


def _write_sharded_checkpoint(
    model_dir: Path, tensors: dict[str, torch.Tensor]
) -> None:
    names = sorted(tensors)
    split = len(names) // 2
    shard_names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    save_file(
        {name: tensors[name] for name in names[:split]},
        model_dir / shard_names[0],
    )
    save_file(
        {name: tensors[name] for name in names[split:]},
        model_dir / shard_names[1],
    )
    weight_map = {
        name: shard_names[0] if index < split else shard_names[1]
        for index, name in enumerate(names)
    }
    total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
        )
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
    @unittest.skipUnless(
        HAS_TRANSFORMERS,
        "the optional Transformers reference dependency is not installed",
    )
    @unittest.skipUnless(
        (_QWEN_MODEL_DIR / "model.safetensors").is_file(),
        "the local Qwen3-0.6B checkpoint is not available",
    )
    def test_formatted_chat_logits_match_transformers_sdpa(self) -> None:
        """Qwen uses the same CPU BF16 operations and should match exactly."""

        tokenizer = Qwen3Tokenizer.from_model_dir(_QWEN_MODEL_DIR)
        input_ids = formatted_input(
            tokenizer,
            "Explain what a KV cache does in one sentence.",
        )

        model = Qwen3ForCausalLM.from_model_dir(_QWEN_MODEL_DIR)
        actual = run_mini_runtime(model, input_ids, generated_tokens=6)
        del model
        gc.collect()
        expected = run_transformers(
            _QWEN_MODEL_DIR,
            input_ids,
            generated_tokens=6,
        )

        torch.testing.assert_close(
            actual.uncached_last_logits,
            expected.uncached_last_logits,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            actual.prefill_logits, expected.prefill_logits, rtol=0.0, atol=0.0
        )
        for ours, reference in zip(
            actual.decode_logits, expected.decode_logits, strict=True
        ):
            torch.testing.assert_close(ours, reference, rtol=0.0, atol=0.0)
        self.assertEqual(actual.token_ids, expected.token_ids)

        our_text = tokenizer.decode(actual.token_ids)
        reference_text = tokenizer.decode(expected.token_ids)
        self.assertEqual(our_text, reference_text)
        print("\nQwen formatted-chat reference comparison")
        print(f"  Our output: {our_text!r}")
        print(f"  HF output:  {reference_text!r}")

    def test_automatic_positions_match_explicit_positions(self) -> None:
        config = _tiny_config()
        model = Qwen3ForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 4, 7]])
        positions = torch.tensor([[0, 1, 2]])

        automatic = model(input_ids)
        explicit = model(input_ids, position_ids=positions)

        torch.testing.assert_close(automatic, explicit)

    def test_module_names_match_checkpoint_contract(self) -> None:
        self.assertEqual(
            set(Qwen3ForCausalLM(_tiny_config()).state_dict()),
            set(expected_qwen3_tensors(_tiny_config())),
        )

    def test_from_model_dir_loads_single_and_sharded_checkpoints(self) -> None:
        config_data = _tiny_config_data(num_hidden_layers=1)
        source = Qwen3ForCausalLM(Qwen3Config.from_dict(config_data)).eval()
        tensors = {name: value.detach() for name, value in source.state_dict().items()}
        input_ids = torch.tensor([[1, 4, 7]])
        expected = source(input_ids)

        for layout in ("single", "sharded"):
            with self.subTest(layout=layout):
                with tempfile.TemporaryDirectory() as directory:
                    model_dir = Path(directory)
                    (model_dir / "config.json").write_text(json.dumps(config_data))
                    if layout == "single":
                        save_file(tensors, model_dir / "model.safetensors")
                    else:
                        _write_sharded_checkpoint(model_dir, tensors)

                    loaded = Qwen3ForCausalLM.from_model_dir(model_dir)
                    actual = loaded(input_ids)

                self.assertFalse(loaded.training)
                self.assertFalse(
                    any(parameter.is_meta for parameter in loaded.parameters())
                )
                torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_token_ids_and_positions(self) -> None:
        model = Qwen3ForCausalLM(_tiny_config())

        with self.assertRaisesRegex(ValueError, "within vocabulary"):
            model(torch.tensor([[32]]))
        with self.assertRaisesRegex(ValueError, "same.*shape"):
            model(torch.tensor([[1, 2]]), position_ids=torch.tensor([[0]]))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model(torch.tensor([[1]]), position_ids=torch.tensor([[-1]]))
        with self.assertRaisesRegex(ValueError, "exceeds the model limit"):
            model(torch.tensor([[1]]), position_ids=torch.tensor([[32]]))


if __name__ == "__main__":
    unittest.main()
