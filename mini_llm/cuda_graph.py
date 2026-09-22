"""Reusable CUDA graph replay for paged-cache decode."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mini_llm.cache.batch import BatchLayerKVCache
from mini_llm.cache.paged import SequenceCacheHandle
from mini_llm.model.base import CausalLMBase


class PagedGraphCache:
    """Bridge fixed graph metadata to a batch of request caches."""

    def __init__(self, caches: Sequence[SequenceCacheHandle]) -> None:
        if not caches:
            raise ValueError("a graph cache batch must contain at least one request")
        self.pool = caches[0].pool
        self.batch_size = len(caches)
        max_blocks = (
            self.pool.max_sequence_length + self.pool.block_size - 1
        ) // self.pool.block_size  # maximum blocks each request can take
        capture_caches: list[SequenceCacheHandle] = []
        for cache in caches:
            block_table = cache.block_table.new_zeros(max_blocks)
            # Capture cache does not contain real K/V data, it's only used to capture the CUDA graph.
            capture_cache = SequenceCacheHandle(
                self.pool,
                self.pool.max_sequence_length,
                block_table,
            )
            capture_caches.append(capture_cache)
        self.layers = [
            BatchLayerKVCache(
                [cache.layers[layer_index] for cache in capture_caches]
            )
            for layer_index in range(self.pool.num_layers)
        ]
        self._block_tables = []
        for layer in self.layers:
            view = layer.view(gathered=False)
            assert view.block_table is not None
            self._block_tables.append(view.block_table)
        self.bind(caches)

    def bind(self, caches: Sequence[SequenceCacheHandle]) -> None:
        """Rebind captured metadata buffers to another compatible request batch.

        CUDA graph replay requires the captured block-table tensor addresses to
        remain fixed. Binding copies each request's current block IDs into those
        existing tensors and records which real caches should receive host-side
        length updates; it does not recapture the graph or move cached K/V data.

        Args:
            caches: Prefilled request caches from the captured pool, in the row
                order expected by the next replay. In order to batch, the
                batch size must match the captured graph. Otherwise we need to
                re-capture the graph with a new batch size.
        """

        caches = tuple(caches)
        if len(caches) != self.batch_size:
            raise ValueError(
                f"graph batch size is {self.batch_size}, got {len(caches)} caches"
            )
        if any(cache.pool is not self.pool for cache in caches):
            raise ValueError("graph caches must use the captured paged pool")
        if any(cache.length == 0 for cache in caches):
            raise ValueError("prefill every cache before binding it to a graph")
        for block_table in self._block_tables:
            block_table.zero_()
            for row, cache in zip(block_table, caches):
                row[: cache.block_table.numel()].copy_(cache.block_table)
        self.requests = caches

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(cache.length for cache in self.requests)

    @property
    def device(self) -> torch.device:
        return self.pool.device

    def advance(self, token_count: int) -> None:
        for cache in self.requests:
            cache.ensure_can_append(token_count)
        for cache in self.requests:
            cache.advance(token_count)

    def rollback(self, lengths: Sequence[int]) -> None:
        for cache, length in zip(self.requests, lengths):
            cache.rollback(length)


class PagedDecodeGraph:
    """Capture fixed-batch decode and rebind compatible paged caches."""

    def __init__(
        self,
        model: CausalLMBase,
        caches: Sequence[SequenceCacheHandle],
    ) -> None:
        if not caches:
            raise ValueError("a decode graph batch must contain at least one cache")
        first = caches[0]
        if model.input_device.type != "cuda" or first.device.type != "cuda":
            raise ValueError("paged decode graphs require a CUDA model and cache")
        if model.input_device != first.device:
            raise ValueError("model and cache must use the same CUDA device")

        self.cache = PagedGraphCache(caches)
        self.input_ids = torch.zeros(
            (self.cache.batch_size, 1), dtype=torch.long, device=first.device
        )
        self.position_ids = torch.tensor(
            self.cache.lengths, dtype=torch.long, device=first.device
        ).unsqueeze(1)

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            hidden_states = model.model(
                self.input_ids,
                position_ids=self.position_ids,
                layer_caches=self.cache.layers,
            )
            self.logits = model._project_logits(hidden_states)

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Decode one token for every bound request and return graph logits."""

        expected_shape = (self.cache.batch_size, 1)
        if input_ids.shape != expected_shape or input_ids.dtype != torch.long:
            raise ValueError(
                f"decode input_ids must be a {expected_shape} torch.long tensor"
            )
        if input_ids.device != self.cache.device:
            raise ValueError("decode input_ids must be on the cache CUDA device")
        self.input_ids.copy_(input_ids)
        previous_lengths = self.cache.lengths
        self.position_ids.copy_(
            torch.tensor(
                previous_lengths,
                dtype=torch.long,
                device=self.cache.device,
            ).unsqueeze(1)
        )
        self.cache.advance(1)
        try:
            self.graph.replay()
        except Exception:
            self.cache.rollback(previous_lengths)
            raise
        return self.logits
