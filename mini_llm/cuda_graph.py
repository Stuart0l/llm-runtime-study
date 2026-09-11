"""CUDA graph replay for one paged-cache decode request."""

from __future__ import annotations

import torch

from mini_llm.cache.paged import SequenceCacheHandle
from mini_llm.model.base import CausalLMBase


class PagedDecodeGraph:
    """Capture and replay full single-token decode for one paged cache."""

    def __init__(self, model: CausalLMBase, cache: SequenceCacheHandle) -> None:
        if model.input_device.type != "cuda" or cache.device.type != "cuda":
            raise ValueError("paged decode graphs require a CUDA model and cache")
        if model.input_device != cache.device:
            raise ValueError("model and cache must use the same CUDA device")
        if cache.length == 0:
            raise ValueError("prefill the cache before capturing decode")

        self.cache = cache
        self.input_ids = torch.zeros((1, 1), dtype=torch.long, device=cache.device)
        self.position_ids = torch.full(
            (1, 1), cache.length, dtype=torch.long, device=cache.device
        )
        start_length = cache.length

        def decode() -> torch.Tensor:
            hidden_states = model.model(
                self.input_ids,
                position_ids=self.position_ids,
                layer_caches=cache.layers,
            )
            return model._project_logits(hidden_states)

        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.logits = decode()
        finally:
            cache.rollback(start_length)

    def replay(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Decode one token and return graph-owned logits."""

        if input_ids.shape != (1, 1) or input_ids.dtype != torch.long:
            raise ValueError("decode input_ids must be a [1, 1] torch.long tensor")
        if input_ids.device != self.cache.device:
            raise ValueError("decode input_ids must be on the cache CUDA device")
        self.input_ids.copy_(input_ids)
        previous_length = self.cache.length
        self.position_ids.fill_(previous_length)
        self.cache.advance(1)
        try:
            self.graph.replay()
        except Exception:
            self.cache.rollback(previous_length)
            raise
        return self.logits
