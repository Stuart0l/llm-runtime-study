from __future__ import annotations

from pathlib import Path
import unittest

from mini_llm.tokenizer import (
    ChatMessage,
    Qwen3Tokenizer,
    TokenizerError,
    format_qwen3_chat,
)
from examples.tokenizer_demo import render_token_mapping


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"


class ChatFormattingTests(unittest.TestCase):
    def test_formats_system_user_and_generation_prompt(self) -> None:
        prompt = format_qwen3_chat(
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("user", "Hello"),
            ],
            enable_thinking=False,
        )

        self.assertEqual(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nHello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_thinking_prompt_ends_after_assistant_header(self) -> None:
        prompt = format_qwen3_chat(
            [ChatMessage("user", "Solve this")], enable_thinking=True
        )

        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", prompt)

    def test_formats_assistant_history(self) -> None:
        prompt = format_qwen3_chat(
            [
                ChatMessage("user", "one"),
                ChatMessage("assistant", "two"),
                ChatMessage("user", "three"),
            ]
        )

        self.assertIn("<|im_start|>assistant\ntwo<|im_end|>\n", prompt)

    def test_can_format_history_without_generation_prompt(self) -> None:
        prompt = format_qwen3_chat(
            [ChatMessage("user", "one"), ChatMessage("assistant", "two")],
            add_generation_prompt=False,
        )

        self.assertTrue(prompt.endswith("two<|im_end|>\n"))

    def test_rejects_invalid_conversation_order(self) -> None:
        with self.assertRaisesRegex(TokenizerError, "expected 'assistant'"):
            format_qwen3_chat(
                [ChatMessage("user", "one"), ChatMessage("user", "two")]
            )

    def test_generation_prompt_requires_final_user_turn(self) -> None:
        with self.assertRaisesRegex(TokenizerError, "final message"):
            format_qwen3_chat(
                [ChatMessage("user", "one"), ChatMessage("assistant", "two")]
            )


@unittest.skipUnless(MODEL_DIR.is_dir(), "local Qwen3 tokenizer is unavailable")
class Qwen3TokenizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = Qwen3Tokenizer.from_model_dir(MODEL_DIR)

    def test_validates_special_token_ids(self) -> None:
        self.assertEqual(self.tokenizer.special_tokens.end_of_text, 151643)
        self.assertEqual(self.tokenizer.special_tokens.im_start, 151644)
        self.assertEqual(self.tokenizer.special_tokens.im_end, 151645)
        self.assertEqual(self.tokenizer.special_tokens.think_start, 151667)
        self.assertEqual(self.tokenizer.special_tokens.think_end, 151668)

    def test_distinguishes_tokenizer_entries_from_model_output_rows(self) -> None:
        self.assertEqual(self.tokenizer.base_vocab_size, 151643)
        self.assertEqual(self.tokenizer.vocab_size, 151669)
        self.assertEqual(self.tokenizer.model_vocab_size, 151936)

    def test_formatted_prompt_round_trips_exactly(self) -> None:
        prompt = format_qwen3_chat([ChatMessage("user", "Hello")])

        token_ids = self.tokenizer.encode(prompt)

        self.assertEqual(self.tokenizer.decode(token_ids), prompt)
        self.assertEqual(token_ids[0], self.tokenizer.special_tokens.im_start)
        self.assertIn(self.tokenizer.special_tokens.im_end, token_ids)

    def test_skip_special_tokens_removes_chat_boundaries(self) -> None:
        prompt = format_qwen3_chat([ChatMessage("user", "Hello")])
        token_ids = self.tokenizer.encode(prompt)

        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        self.assertNotIn("<|im_start|>", decoded)
        self.assertNotIn("<|im_end|>", decoded)
        self.assertIn("<think>", decoded)

    def test_demo_renders_one_to_one_token_id_mapping(self) -> None:
        token_ids = self.tokenizer.encode("<|im_start|>user\nHello<|im_end|>")

        mapping = render_token_mapping(self.tokenizer, token_ids)

        self.assertIn("    0  '<|im_start|>'              151644", mapping)
        self.assertIn("    1  'user'                         872", mapping)
        self.assertIn("    2  'Ċ'                            198", mapping)
        self.assertIn("    3  'Hello'                       9707", mapping)
        self.assertIn("    4  '<|im_end|>'                151645", mapping)


if __name__ == "__main__":
    unittest.main()
