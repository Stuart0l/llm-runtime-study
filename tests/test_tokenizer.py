from __future__ import annotations

from datetime import date
from pathlib import Path
import unittest

from mini_llm.chat import GraniteChatTemplate, Qwen3ChatTemplate
from mini_llm.tokenizer import (
    ChatMessage,
    GraniteTokenizer,
    Qwen3Tokenizer,
    TokenizerError,
    load_tokenizer,
)
from examples.tokenizer_demo import render_token_mapping


MODEL_DIR = Path(__file__).parents[1] / "models" / "qwen3-0.6b"
GRANITE_MODEL_DIR = Path(__file__).parents[1] / "models" / "granite-3.1-1b"


class ChatFormattingTests(unittest.TestCase):
    def test_formats_system_user_and_generation_prompt(self) -> None:
        prompt = Qwen3ChatTemplate.format(
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
        prompt = Qwen3ChatTemplate.format(
            [ChatMessage("user", "Solve this")],
            enable_thinking=True,
        )

        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", prompt)

    def test_formats_assistant_history(self) -> None:
        prompt = Qwen3ChatTemplate.format(
            [
                ChatMessage("user", "one"),
                ChatMessage("assistant", "two"),
                ChatMessage("user", "three"),
            ],
        )

        self.assertIn("<|im_start|>assistant\ntwo<|im_end|>\n", prompt)

    def test_can_format_history_without_generation_prompt(self) -> None:
        prompt = Qwen3ChatTemplate.format(
            [ChatMessage("user", "one"), ChatMessage("assistant", "two")],
            add_generation_prompt=False,
        )

        self.assertTrue(prompt.endswith("two<|im_end|>\n"))

    def test_rejects_invalid_conversation_order(self) -> None:
        with self.assertRaisesRegex(TokenizerError, "expected 'assistant'"):
            Qwen3ChatTemplate.format(
                [ChatMessage("user", "one"), ChatMessage("user", "two")],
            )

    def test_generation_prompt_requires_final_user_turn(self) -> None:
        with self.assertRaisesRegex(TokenizerError, "final message"):
            Qwen3ChatTemplate.format(
                [ChatMessage("user", "one"), ChatMessage("assistant", "two")],
            )


class GraniteChatFormattingTests(unittest.TestCase):
    def test_formats_supplied_system_message_and_generation_prompt(self) -> None:
        prompt = GraniteChatTemplate.format(
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("user", "Hello"),
            ],
        )

        self.assertEqual(
            prompt,
            "<|start_of_role|>system<|end_of_role|>Be concise.<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n"
            "<|start_of_role|>assistant<|end_of_role|>",
        )

    def test_injects_official_dated_default_system_message(self) -> None:
        prompt = GraniteChatTemplate.format(
            [ChatMessage("user", "Hello")],
            today=date(2026, 8, 28),
        )

        self.assertTrue(
            prompt.startswith(
                "<|start_of_role|>system<|end_of_role|>"
                "Knowledge Cutoff Date: April 2024.\n"
                "Today's Date: August 28, 2026.\n"
                "You are Granite, developed by IBM. "
                "You are a helpful AI assistant.<|end_of_text|>\n"
            )
        )

    def test_formats_history_without_duplicating_supplied_system_turn(self) -> None:
        prompt = GraniteChatTemplate.format(
            [
                ChatMessage("system", "Rules"),
                ChatMessage("user", "one"),
                ChatMessage("assistant", "two"),
            ],
            add_generation_prompt=False,
        )

        self.assertEqual(prompt.count("<|start_of_role|>system"), 1)
        self.assertTrue(prompt.endswith("two<|end_of_text|>\n"))

    def test_rejects_qwen_thinking_mode(self) -> None:
        with self.assertRaisesRegex(TokenizerError, "not Granite"):
            GraniteChatTemplate.format(
                [ChatMessage("user", "Hello")],
                enable_thinking=True,
            )


class ChatTemplateRegistrationTests(unittest.TestCase):
    def test_each_tokenizer_declares_its_architecture_template(self) -> None:
        self.assertIs(Qwen3Tokenizer.chat_template, Qwen3ChatTemplate)
        self.assertIs(GraniteTokenizer.chat_template, GraniteChatTemplate)


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
        prompt = self.tokenizer.format_chat([ChatMessage("user", "Hello")])

        token_ids = self.tokenizer.encode(prompt)

        self.assertEqual(self.tokenizer.decode(token_ids), prompt)
        self.assertEqual(token_ids[0], self.tokenizer.special_tokens.im_start)
        self.assertIn(self.tokenizer.special_tokens.im_end, token_ids)

    def test_skip_special_tokens_removes_chat_boundaries(self) -> None:
        prompt = self.tokenizer.format_chat([ChatMessage("user", "Hello")])
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


@unittest.skipUnless(
    GRANITE_MODEL_DIR.is_dir(), "local Granite tokenizer is unavailable"
)
class GraniteTokenizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = GraniteTokenizer.from_model_dir(GRANITE_MODEL_DIR)

    def test_factory_dispatches_to_granite_tokenizer(self) -> None:
        tokenizer = load_tokenizer(GRANITE_MODEL_DIR)

        self.assertIsInstance(tokenizer, GraniteTokenizer)

    def test_validates_official_special_token_ids(self) -> None:
        self.assertEqual(self.tokenizer.special_tokens.end_of_text, 0)
        self.assertEqual(self.tokenizer.special_tokens.start_of_role, 49152)
        self.assertEqual(self.tokenizer.special_tokens.end_of_role, 49153)
        self.assertEqual(self.tokenizer.special_tokens.tool_call, 49154)

    def test_tokenizer_and_model_vocabulary_sizes_match(self) -> None:
        self.assertEqual(self.tokenizer.base_vocab_size, 49152)
        self.assertEqual(self.tokenizer.vocab_size, 49155)
        self.assertEqual(self.tokenizer.model_vocab_size, 49155)

    def test_formatted_prompt_round_trips_exactly(self) -> None:
        prompt = self.tokenizer.format_chat(
            [
                ChatMessage("system", "Be concise."),
                ChatMessage("user", "Hello"),
            ]
        )
        token_ids = self.tokenizer.encode(prompt)

        self.assertEqual(self.tokenizer.decode(token_ids), prompt)
        self.assertEqual(token_ids[0], self.tokenizer.special_tokens.start_of_role)
        self.assertIn(self.tokenizer.special_tokens.end_of_text, token_ids)

    def test_prompt_text_and_ids_match_transformers(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("Transformers reference runtime is unavailable")

        messages = [
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "Hello"),
        ]
        reference_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        reference = AutoTokenizer.from_pretrained(
            GRANITE_MODEL_DIR, local_files_only=True
        )

        our_prompt = self.tokenizer.format_chat(messages)
        reference_prompt = reference.apply_chat_template(
            reference_messages, tokenize=False, add_generation_prompt=True
        )
        reference_ids = reference.apply_chat_template(
            reference_messages, tokenize=True, add_generation_prompt=True
        )

        self.assertEqual(our_prompt, reference_prompt)
        self.assertEqual(self.tokenizer.encode(our_prompt), reference_ids)


if __name__ == "__main__":
    unittest.main()
