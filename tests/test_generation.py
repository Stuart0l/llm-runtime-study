from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import torch
from torch import nn

from mini_llm.generation import GenerationError, generate
from mini_llm.sampling import (
    SamplingConfig,
    SamplingError,
    filter_logits,
    make_generator,
    sample_next_token,
)
from mini_llm.tokenizer import ChatMessage, TokenizerError, format_qwen3_chat


class SamplingTests(unittest.TestCase):
    def test_temperature_zero_is_greedy(self) -> None:
        token = sample_next_token(
            torch.tensor([-4.0, 7.0, 2.0]), SamplingConfig(temperature=0)
        )
        self.assertEqual(token, 1)

    def test_temperature_rescales_logits(self) -> None:
        filtered = filter_logits(
            torch.tensor([1.0, 2.0, 3.0]), SamplingConfig(temperature=0.5)
        )
        torch.testing.assert_close(filtered, torch.tensor([2.0, 4.0, 6.0]))

    def test_top_k_removes_all_but_k_largest_logits(self) -> None:
        filtered = filter_logits(
            torch.tensor([1.0, 4.0, 2.0, 3.0]),
            SamplingConfig(temperature=1, top_k=2),
        )
        self.assertTrue(torch.isneginf(filtered[[0, 2]]).all().item())
        torch.testing.assert_close(filtered[[1, 3]], torch.tensor([4.0, 3.0]))

    def test_top_p_keeps_smallest_prefix_reaching_probability_mass(self) -> None:
        logits = torch.log(torch.tensor([0.60, 0.25, 0.10, 0.05]))
        filtered = filter_logits(
            logits, SamplingConfig(temperature=1, top_p=0.70)
        )
        self.assertTrue(torch.isfinite(filtered[:2]).all().item())
        self.assertTrue(torch.isneginf(filtered[2:]).all().item())

    def test_same_seed_produces_same_random_sequence(self) -> None:
        logits = torch.tensor([0.0, 0.2, 0.4, 0.6])
        config = SamplingConfig(temperature=0.8, seed=123)
        first_generator = make_generator(config.seed)
        second_generator = make_generator(config.seed)

        first = [
            sample_next_token(logits, config, generator=first_generator)
            for _ in range(20)
        ]
        second = [
            sample_next_token(logits, config, generator=second_generator)
            for _ in range(20)
        ]
        self.assertEqual(first, second)

    def test_rejects_invalid_sampling_parameters_and_logits(self) -> None:
        with self.assertRaises(SamplingError):
            SamplingConfig(temperature=-1)
        with self.assertRaises(SamplingError):
            SamplingConfig(top_k=0)
        with self.assertRaises(SamplingError):
            SamplingConfig(top_p=0)
        with self.assertRaisesRegex(SamplingError, "vocabulary size"):
            sample_next_token(
                torch.ones(3), SamplingConfig(temperature=1, top_k=4)
            )


class _FakeTokenizer:
    pieces = {2: "Hello", 3: " world", 4: ""}

    def __init__(self, prompt_ids: list[int] | None = None) -> None:
        self.prompt_ids = [0, 1] if prompt_ids is None else prompt_ids
        self.encoded_text: str | None = None
        self.decoded_token_ids: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        self.encoded_text = text
        return self.prompt_ids

    def format_chat(self, messages, *, enable_thinking=False) -> str:
        return format_qwen3_chat(messages, enable_thinking=enable_thinking)

    def decode(self, token_ids, *, skip_special_tokens=False) -> str:
        self.decoded_token_ids.append(list(token_ids))
        return "".join(self.pieces[token_id] for token_id in token_ids)


class _SplitUnicodeTokenizer:
    def format_chat(self, messages, *, enable_thinking=False) -> str:
        return format_qwen3_chat(messages, enable_thinking=enable_thinking)

    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, token_ids, *, skip_special_tokens=False) -> str:
        if token_ids == [2]:
            return "\ufffd"
        if token_ids[:2] == [2, 3]:
            return "你"
        return ""


class _FakeModel(nn.Module):
    def __init__(self, next_tokens: list[int], *, context_limit: int = 8) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0), requires_grad=False)
        self.model = SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=self.anchor)
        )
        self.config = SimpleNamespace(
            vocab_size=5,
            max_position_embeddings=context_limit,
            eos_token_ids=(4,),
        )
        self.next_tokens = next_tokens
        self.prefill_calls = 0
        self.decode_inputs: list[int] = []
        self.cache_capacity: int | None = None

    @property
    def input_device(self) -> torch.device:
        return self.anchor.device

    def setup_cache(self, capacity: int) -> None:
        self.cache_capacity = capacity

    def _logits(self, next_token: int, sequence_length: int) -> torch.Tensor:
        logits = torch.full((1, sequence_length, 5), -10.0)
        logits[0, -1, next_token] = 10.0
        return logits

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.prefill_calls += 1
        return self._logits(self.next_tokens[0], input_ids.shape[1])

    def decode(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.decode_inputs.append(int(input_ids.item()))
        return self._logits(self.next_tokens[len(self.decode_inputs)], 1)


class GenerationTests(unittest.TestCase):
    def test_formats_complete_chat_history_before_iteration(self) -> None:
        model = _FakeModel([2, 4])
        tokenizer = _FakeTokenizer()
        messages = [
            ChatMessage("system", "Be concise."),
            ChatMessage("user", "What is a KV cache?"),
            ChatMessage("assistant", "Stored attention keys and values."),
            ChatMessage("user", "Why keep it?"),
        ]

        stream = generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=2,
            sampling=SamplingConfig(temperature=0),
        )

        self.assertEqual(model.prefill_calls, 0)
        self.assertEqual(
            tokenizer.encoded_text,
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nWhat is a KV cache?<|im_end|>\n"
            "<|im_start|>assistant\nStored attention keys and values.<|im_end|>\n"
            "<|im_start|>user\nWhy keep it?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

        events = list(stream)

        self.assertEqual(model.prefill_calls, 1)
        self.assertEqual([event.token_id for event in events], [2, 4])

    def test_rejects_invalid_chat_history_eagerly(self) -> None:
        model = _FakeModel([2])

        with self.assertRaisesRegex(TokenizerError, "expected 'assistant'"):
            generate(
                model,
                _FakeTokenizer(),
                [ChatMessage("user", "one"), ChatMessage("user", "two")],
                max_new_tokens=1,
            )

    def test_uses_one_prefill_then_single_token_decode_and_stops_at_eos(self) -> None:
        model = _FakeModel([2, 3, 4])
        tokenizer = _FakeTokenizer()

        events = list(
            generate(
                model,
                tokenizer,
                [ChatMessage("user", "Say hello")],
                max_new_tokens=6,
                sampling=SamplingConfig(temperature=0),
            )
        )

        self.assertEqual(model.prefill_calls, 1)
        self.assertEqual(model.decode_inputs, [2, 3])
        self.assertEqual(model.cache_capacity, 8)
        self.assertEqual([event.token_id for event in events], [2, 3, 4])
        self.assertEqual([event.text_delta for event in events], ["Hello", " world", ""])
        self.assertEqual(events[-1].text, "Hello world")
        self.assertEqual(events[-1].finish_reason, "eos")
        self.assertEqual(events[0].prompt_token_count, 2)
        self.assertIsNone(events[1].prompt_token_count)
        self.assertEqual(tokenizer.decoded_token_ids, [[2], [3], [4]])
        self.assertEqual(
            tokenizer.encoded_text,
            "<|im_start|>user\nSay hello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_stops_at_requested_token_limit(self) -> None:
        model = _FakeModel([2, 3, 2])
        events = list(
            generate(
                model,
                _FakeTokenizer(prompt_ids=[0]),
                [ChatMessage("user", "question")],
                max_new_tokens=2,
            )
        )

        self.assertEqual([event.token_id for event in events], [2, 3])
        self.assertEqual(model.decode_inputs, [2])
        self.assertEqual(events[-1].finish_reason, "max_new_tokens")

    def test_buffers_incomplete_unicode_before_emitting_text_delta(self) -> None:
        model = _FakeModel([2, 3, 4])
        events = list(
            generate(
                model,
                _SplitUnicodeTokenizer(),
                [ChatMessage("user", "question")],
                max_new_tokens=3,
            )
        )

        self.assertEqual([event.text_delta for event in events], ["", "你", ""])
        self.assertEqual(events[-1].text, "你")

    def test_stream_yield_does_not_leak_inference_mode_to_caller(self) -> None:
        stream = generate(
            _FakeModel([2, 3]),
            _FakeTokenizer(prompt_ids=[0]),
            [ChatMessage("user", "question")],
            max_new_tokens=2,
        )

        next(stream)

        self.assertFalse(torch.is_inference_mode_enabled())

    def test_synchronized_model_timings_are_attached_to_events(self) -> None:
        synchronize = MagicMock()
        with patch(
            "mini_llm.generation.time.perf_counter",
            side_effect=[1.0, 1.2, 2.0, 2.1, 3.0, 3.05],
        ):
            events = list(
                generate(
                    _FakeModel([2, 3, 4]),
                    _FakeTokenizer(prompt_ids=[0]),
                    [ChatMessage("user", "question")],
                    max_new_tokens=3,
                    synchronize=synchronize,
                )
            )

        self.assertEqual(synchronize.call_count, 6)
        self.assertAlmostEqual(events[0].model_seconds, 0.2)
        self.assertAlmostEqual(events[1].model_seconds, 0.1)
        self.assertAlmostEqual(events[2].model_seconds, 0.05)

    def test_stops_before_exceeding_context_limit(self) -> None:
        model = _FakeModel([2, 3], context_limit=3)
        events = list(
            generate(
                model,
                _FakeTokenizer(),
                [ChatMessage("user", "question")],
                max_new_tokens=5,
            )
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].token_id, 2)
        self.assertEqual(events[0].finish_reason, "context_length")
        self.assertEqual(model.cache_capacity, 3)
        self.assertEqual(model.decode_inputs, [])

    def test_full_prompt_returns_terminal_context_event_without_forward(self) -> None:
        model = _FakeModel([2], context_limit=2)
        events = list(
            generate(
                model,
                _FakeTokenizer(),
                [ChatMessage("user", "question")],
                max_new_tokens=1,
            )
        )

        self.assertEqual(model.prefill_calls, 0)
        self.assertIsNone(events[0].token_id)
        self.assertEqual(events[0].finish_reason, "context_length")
        self.assertEqual(events[0].prompt_token_count, 2)

    def test_rejects_empty_history_and_non_positive_token_limit(self) -> None:
        model = _FakeModel([2])
        with self.assertRaisesRegex(TokenizerError, "at least one"):
            generate(model, _FakeTokenizer(), [], max_new_tokens=1)
        with self.assertRaisesRegex(GenerationError, "positive"):
            generate(
                model,
                _FakeTokenizer(),
                [ChatMessage("user", "question")],
                max_new_tokens=0,
            )


if __name__ == "__main__":
    unittest.main()
