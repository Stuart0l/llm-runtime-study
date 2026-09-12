"""Padded attention views over independent request caches."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from mini_llm.cache.contracts import LayerKVCache, LayerKVCacheView


class PaddedBatchLayerKVCache:
    """Combine one layer from each request into a padded SDPA cache."""

    def __init__(self, caches: Sequence[LayerKVCache]) -> None:
        if not caches:
            raise ValueError("a cache batch must contain at least one request")
        self.caches = tuple(caches)

    @property
    def length(self) -> int:
        return max(cache.length for cache in self.caches)

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        if keys.shape[0] != len(self.caches):
            raise ValueError(
                f"K/V batch size {keys.shape[0]} does not match "
                f"{len(self.caches)} caches"
            )
        for index, cache in enumerate(self.caches):
            cache.write(
                keys[index : index + 1],
                values[index : index + 1],
                position_ids[index : index + 1],
            )

    def view(self, *, gathered: bool) -> LayerKVCacheView:
        if not gathered:
            raise ValueError("a padded cache batch only exposes gathered K/V")
        views = [cache.view(gathered=True) for cache in self.caches]
        length = max(view.keys.shape[2] for view in views)

        def pad(tensor: torch.Tensor) -> torch.Tensor:
            return F.pad(tensor, (0, 0, 0, length - tensor.shape[2]))

        return LayerKVCacheView(
            keys=torch.cat([pad(view.keys) for view in views]),
            values=torch.cat([pad(view.values) for view in views]),
            block_table=None,
            capacity=length,
        )

    def append(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.write(keys, values, position_ids)
        view = self.view(gathered=True)
        return view.keys, view.values
