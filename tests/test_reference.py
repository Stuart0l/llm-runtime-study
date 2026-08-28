from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import unittest

import torch

from mini_llm.model import Qwen3ForCausalLM
from mini_llm.tokenizer import ChatMessage, Qwen3Tokenizer, format_qwen3_chat


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _PROJECT_ROOT / "models" / "qwen3-0.6b"
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(
    _HAS_TRANSFORMERS,
    "the optional Transformers reference dependency is not installed",
)
@unittest.skipUnless(
    (_MODEL_DIR / "model.safetensors").is_file(),
    "the local Qwen3-0.6B checkpoint is not available",
)
class TransformersReferenceTests(unittest.TestCase):
    def test_formatted_chat_logits_match_transformers_sdpa(self) -> None:
        """Compare our complete model with an independent reference runtime.

        Both implementations receive identical IDs and use CPU BF16 SDPA.  The
        models are held in memory sequentially rather than simultaneously so
        this optional correctness test remains practical on a development Mac.
        """

        # Import only inside the optional test: Transformers must never become
        # a dependency of the mini runtime's actual loading or forward path.
        from transformers import AutoModelForCausalLM

        tokenizer = Qwen3Tokenizer.from_model_dir(_MODEL_DIR)
        prompt = format_qwen3_chat(
            [
                ChatMessage(
                    role="user",
                    content="Explain what a KV cache does in one sentence.",
                )
            ],
            enable_thinking=False,
        )
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        model = Qwen3ForCausalLM.from_model_dir(_MODEL_DIR)
        model.setup_cache(input_ids.shape[1] + 6)
        with torch.inference_mode():
            actual = model(input_ids).float().clone()
            cached_prefill = model.prefill(input_ids).float()
            our_tokens = [int(cached_prefill[0, -1].argmax().item())]
            our_decode_logits = []
            for _ in range(5):
                token_input = torch.tensor([[our_tokens[-1]]], dtype=torch.long)
                decode_logits = model.decode(token_input).float().clone()
                our_decode_logits.append(decode_logits)
                next_token = int(decode_logits[0, -1].argmax().item())
                our_tokens.append(next_token)
        del model
        gc.collect()

        reference = AutoModelForCausalLM.from_pretrained(
            _MODEL_DIR,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
        with torch.inference_mode():
            reference_prefill = reference(input_ids=input_ids, use_cache=True)
            expected = reference_prefill.logits.float()
            reference_tokens = [int(expected[0, -1].argmax().item())]
            past_key_values = reference_prefill.past_key_values
            reference_decode_logits = []
            for _ in range(5):
                reference_decode = reference(
                    input_ids=torch.tensor(
                        [[reference_tokens[-1]]], dtype=torch.long
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = reference_decode.past_key_values
                decode_logits = reference_decode.logits.float()
                reference_decode_logits.append(decode_logits)
                next_token = int(decode_logits[0, -1].argmax().item())
                reference_tokens.append(next_token)

        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(cached_prefill, expected, rtol=0.0, atol=0.0)
        for our_decode, reference_decode in zip(
            our_decode_logits, reference_decode_logits, strict=True
        ):
            torch.testing.assert_close(
                our_decode, reference_decode, rtol=0.0, atol=0.0
            )
        torch.testing.assert_close(
            actual.argmax(dim=-1), expected.argmax(dim=-1), rtol=0.0, atol=0.0
        )
        self.assertEqual(our_tokens, reference_tokens)

        our_text = tokenizer.decode(our_tokens)
        reference_text = tokenizer.decode(reference_tokens)
        self.assertEqual(our_text, reference_text)
        print("\nFormatted-chat reference comparison")
        print(f"  Our token IDs: {our_tokens}")
        print(f"  HF token IDs:  {reference_tokens}")
        print(f"  Our output:    {our_text!r}")
        print(f"  HF output:     {reference_text!r}")


if __name__ == "__main__":
    unittest.main()
