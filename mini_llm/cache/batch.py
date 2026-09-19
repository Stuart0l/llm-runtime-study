"""Attention views over independent request caches."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mini_llm.cache.contracts import LayerKVCache, LayerKVCacheView


class BatchLayerKVCache:
    """One layer KV cache for a batch of independent requests."""

    def __init__(self, caches: Sequence[LayerKVCache]) -> None:
        """Create a batch from requests' caches of the same decoder layer.

        Args:
            caches: Per-request layer caches in batch order.
        """

        if not caches:
            raise ValueError("a cache batch must contain at least one request")
        self.caches = tuple(caches)
        self._block_table: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """Return the longest logical sequence length in the batch."""

        return max(cache.length for cache in self.caches)

    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Write new states into every request cache.

        Args:
            keys: Key states shaped ``[batch, kv_heads, tokens, head_dim]``.
            values: Value states with the same shape as ``keys``.
            position_ids: Logical cache positions shaped ``[batch, tokens]``.
        """

        if keys.shape[0] != len(self.caches):
            raise ValueError(
                f"K/V batch size {keys.shape[0]} does not match "
                f"{len(self.caches)} caches"
            )
        from mini_llm.cache.paged import PagedLayerKVCache

        if all(isinstance(cache, PagedLayerKVCache) for cache in self.caches):
            first = self.caches[0]
            pool = first.handle.pool
            if any(
                cache.handle.pool is not pool
                or cache.layer_index != first.layer_index
                for cache in self.caches
            ):
                raise ValueError(
                    "a paged cache batch must share one pool and layer"
                )
            token_count = keys.shape[2]
            expected = (
                len(self.caches),
                pool.num_key_value_heads,
                token_count,
                pool.head_dim,
            )
            if tuple(keys.shape) != expected or tuple(values.shape) != expected:
                raise ValueError(
                    f"new K/V must both have shape {expected}, got "
                    f"{tuple(keys.shape)} and {tuple(values.shape)}"
                )
            if keys.dtype != pool.dtype or values.dtype != pool.dtype:
                raise ValueError(f"new K/V dtype must match cache dtype {pool.dtype}")
            if keys.device != pool.device or values.device != pool.device:
                raise ValueError(
                    f"new K/V device must match cache device {pool.device}"
                )
            ends = [cache.length + token_count for cache in self.caches]
            if any(
                end > cache.handle.capacity
                for cache, end in zip(self.caches, ends)
            ):
                raise ValueError("KV cache capacity exceeded")

            cache_view = self.view(gathered=False)  # get the block table for the batch
            assert cache_view.block_table is not None
            # Map every logical token position to its request's physical slot
            # in the shared block pool.
            table_indices = torch.div(
                position_ids, pool.block_size, rounding_mode="floor"
            )
            block_offsets = torch.remainder(position_ids, pool.block_size)
            # Get the physical block id for each new token position in the batch.
            block_ids = cache_view.block_table.gather(1, table_indices)
            source_keys = keys.transpose(1, 2)
            source_values = values.transpose(1, 2)
            with torch.no_grad():
                # Address the block and its token offset directly while still
                # writing the whole request batch with one operation per K/V.
                pool.keys[first.layer_index].index_put_(
                    (block_ids, block_offsets), source_keys
                )
                pool.values[first.layer_index].index_put_(
                    (block_ids, block_offsets), source_values
                )
            # Device writes do not update the host-side logical cache lengths.
            for cache, end in zip(self.caches, ends):
                cache.handle._layer_lengths[first.layer_index] = end
            return

        for index, cache in enumerate(self.caches):
            cache.write(
                keys[index : index + 1],
                values[index : index + 1],
                position_ids[index : index + 1],
            )

    def view(self, *, gathered: bool) -> LayerKVCacheView:
        """Return the batched representation consumed by attention.

        Args:
            gathered: If true, return zero-padded contiguous K/V tensors. If
                false, return the shared physical paged pool and a padded block
                table without gathering K/V states.
        """

        views = [cache.view(gathered=gathered) for cache in self.caches]
        if not gathered:
            # Paged cache, share one KV pool
            keys = views[0].keys  # a ref to all keys in the shared pool
            values = views[0].values  # a ref to all values in the shared pool
            if any(
                view.block_table is None
                or view.keys is not keys
                or view.values is not values
                for view in views
            ):
                raise ValueError(
                    "a scattered cache batch must share one paged K/V pool"
                )
            if self._block_table is None:
                block_count = max(view.block_table.shape[1] for view in views)
                self._block_table = views[0].block_table.new_zeros(
                    (len(views), block_count)
                )
                for index, view in enumerate(views):
                    self._block_table[index, : view.block_table.shape[1]].copy_(
                        view.block_table[0]
                    )
            return LayerKVCacheView(
                keys=keys,
                values=values,
                # Entries beyond each request's length are ignored, so block
                # zero is safe padding for shorter tables.
                block_table=self._block_table,
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
        """Write new states and return zero-padded contiguous K/V tensors.

        Args:
            keys: Key states shaped ``[batch, kv_heads, tokens, head_dim]``.
            values: Value states with the same shape as ``keys``.
            position_ids: Logical cache positions shaped ``[batch, tokens]``.

        Returns:
            The gathered key and value tensors, each padded to the longest
            sequence in the batch.
        """

        self.write(keys, values, position_ids)
        view = self.view(gathered=True)
        return view.keys, view.values
