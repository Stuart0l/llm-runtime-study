"""Reusable CUDA graph replay for paged-cache decode."""

from __future__ import annotations

import math

from collections.abc import Sequence

import torch

from mini_llm.cache.paged import PagedBatchKVCache, PagedSequenceKVCache
from mini_llm.model.base import CausalLMBase


class PagedGraphCache:
    """Bridge fixed graph metadata to a batch of request caches."""

    def __init__(
        self, caches: Sequence[PagedSequenceKVCache], max_blocks: int
    ) -> None:
        if not caches:
            raise ValueError("a graph cache batch must contain at least one request")
        self.store = caches[0].store
        self.batch_size = len(caches)
        # Capture must see the widest block table a request can ever use, so
        # every later bind fits into the tensors the graph recorded.
        self.batch = PagedBatchKVCache(
            self.store, caches, block_table_capacity=max_blocks
        )
        self.layers = self.batch.layers
        self.bind(caches)

    def bind(self, caches: Sequence[PagedSequenceKVCache]) -> None:
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
        if any(cache.store is not self.store for cache in caches):
            raise ValueError("graph caches must use the captured paged store")
        if any(cache.length == 0 for cache in caches):
            raise ValueError("prefill every cache before binding it to a graph")
        self.batch.rebind(caches)
        self.requests = caches

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(cache.length for cache in self.requests)

    @property
    def device(self) -> torch.device:
        return self.store.spec.device

    def extend(self, token_count: int) -> None:
        previous_lengths = self.lengths
        extended: list[PagedSequenceKVCache] = []
        try:
            for cache in self.requests:
                cache.extend(token_count)
                extended.append(cache)
        except Exception:
            for cache, length in zip(extended, previous_lengths):
                cache.rollback(length)
            raise

    def rollback(self, lengths: Sequence[int]) -> None:
        for cache, length in zip(self.requests, lengths):
            cache.rollback(length)


class PagedDecodeGraph:
    """Capture fixed-batch decode and rebind compatible paged caches."""

    def __init__(
        self,
        model: CausalLMBase,
        caches: Sequence[PagedSequenceKVCache],
    ) -> None:
        if not caches:
            raise ValueError("a decode graph batch must contain at least one cache")
        first = caches[0]
        store = first.store
        device = store.spec.device
        if model.input_device.type != "cuda" or device.type != "cuda":
            raise ValueError("paged decode graphs require a CUDA model and cache")
        if model.input_device != device:
            raise ValueError("model and cache must use the same CUDA device")

        max_blocks = math.ceil(
            model.config.max_position_embeddings / store.block_size
        )
        self.cache = PagedGraphCache(caches, max_blocks)
        self.input_ids = torch.zeros(
            (self.cache.batch_size, 1), dtype=torch.long, device=device
        )
        self.position_ids = torch.tensor(
            self.cache.lengths, dtype=torch.long, device=device
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
        self.cache.extend(1)
        try:
            self.graph.replay()
        except Exception:
            self.cache.rollback(previous_lengths)
            raise
        return self.logits
