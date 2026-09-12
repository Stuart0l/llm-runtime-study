"""Reusable CUDA graph replay for paged-cache decode."""

from __future__ import annotations

import torch

from mini_llm.cache.paged import SequenceCacheHandle
from mini_llm.model.base import CausalLMBase


class PagedGraphCache:
    """Bridge fixed graph inputs to the currently bound request cache."""

    def __init__(self, cache: SequenceCacheHandle) -> None:
        self.pool = cache.pool
        self.block_table = cache.block_table.new_zeros(
            (self.pool.max_sequence_length + self.pool.block_size - 1)
            // self.pool.block_size
        )
        capture_cache = SequenceCacheHandle(
            self.pool,
            self.pool.max_sequence_length,
            self.block_table,
        )
        capture_cache.advance(cache.length)
        self.layers = capture_cache.layers
        self.bind(cache)

    def bind(self, cache: SequenceCacheHandle) -> None:
        if cache.pool is not self.pool:
            raise ValueError("graph cache and request cache must use the same pool")
        self.block_table[: cache.block_table.numel()].copy_(cache.block_table)
        self.request = cache

    @property
    def length(self) -> int:
        return self.request.length

    @property
    def device(self) -> torch.device:
        return self.request.device

    def advance(self, token_count: int) -> None:
        self.request.advance(token_count)

    def rollback(self, length: int) -> None:
        self.request.rollback(length)


class PagedDecodeGraph:
    """Capture single-token decode and rebind it to compatible paged caches."""

    def __init__(self, model: CausalLMBase, cache: SequenceCacheHandle) -> None:
        if model.input_device.type != "cuda" or cache.device.type != "cuda":
            raise ValueError("paged decode graphs require a CUDA model and cache")
        if model.input_device != cache.device:
            raise ValueError("model and cache must use the same CUDA device")
        if cache.length == 0:
            raise ValueError("prefill the cache before capturing decode")

        self.cache = PagedGraphCache(cache)
        self.input_ids = torch.zeros((1, 1), dtype=torch.long, device=cache.device)
        self.position_ids = torch.full(
            (1, 1), cache.length, dtype=torch.long, device=cache.device
        )

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            hidden_states = model.model(
                self.input_ids,
                position_ids=self.position_ids,
                layer_caches=self.cache.layers,
            )
            self.logits = model._project_logits(hidden_states)

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Decode one token and return graph-owned logits."""

        if input_ids.shape != (1, 1) or input_ids.dtype != torch.long:
            raise ValueError("decode input_ids must be a [1, 1] torch.long tensor")
        if input_ids.device != self.cache.device:
            raise ValueError("decode input_ids must be on the cache CUDA device")
        self.input_ids.copy_(input_ids)
        previous_length = self.cache.length
        self.position_ids.fill_(previous_length)
        # `advance` is required to calculate the correct position_ids for the next token
        self.cache.advance(1)
        try:
            self.graph.replay()
        except Exception:
            self.cache.rollback(previous_length)
            raise
        return self.logits
