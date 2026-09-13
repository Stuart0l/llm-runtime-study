"""Attention views over independent request caches."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mini_llm.cache.contracts import LayerKVCache, LayerKVCacheView


class BatchLayerKVCache:
    """Combine one layer from each request for batched attention."""

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
        views = [cache.view(gathered=gathered) for cache in self.caches]
        if not gathered:
            # Paged cache, share one KV pool
            keys = views[0].keys
            values = views[0].values
            if any(
                view.block_table is None
                or view.keys is not keys
                or view.values is not values
                for view in views
            ):
                raise ValueError(
                    "a scattered cache batch must share one paged K/V pool"
                )
            block_count = max(view.block_table.shape[1] for view in views)
            block_table = views[0].block_table.new_zeros(
                (len(views), block_count)
            )
            for index, view in enumerate(views):
                block_table[index, : view.block_table.shape[1]].copy_(
                    view.block_table[0]
                )
            return LayerKVCacheView(
                keys=keys,
                values=values,
                # Entries beyond each request's length are ignored, so block
                # zero is safe padding for shorter tables.
                block_table=block_table,
                capacity=max(view.capacity for view in views),
            )

        # Dense cache, copy K/V directly into one zero-padded batch allocation.
        length = max(view.keys.shape[2] for view in views)
        _, heads, _, head_dim = views[0].keys.shape
        batch_shape = (len(views), heads, length, head_dim)
        keys = views[0].keys.new_zeros(batch_shape)
        values = views[0].values.new_zeros(batch_shape)
        for batch_index, view in enumerate(views):
            request_length = view.keys.shape[2]
            keys[batch_index, :, :request_length].copy_(view.keys[0])
            values[batch_index, :, :request_length].copy_(view.values[0])

        return LayerKVCacheView(
            keys=keys,
            values=values,
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
