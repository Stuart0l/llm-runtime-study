"""Block-paged K/V tensor storage shared by every request."""

from __future__ import annotations

import torch

from mini_llm.cache.contracts import GatheredKV, KVCacheError, KVCacheSpec


class PagedKVStore:
    """Per-layer K/V blocks addressed by physical block ID.

    Blocks use layout ``[blocks, block_size, kv_heads, head_dim]``; one block ID
    selects the same slot in every decoder layer.
    """

    __slots__ = ("spec", "num_blocks", "block_size", "keys", "values")

    def __init__(self, spec: KVCacheSpec, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise KVCacheError(f"num_blocks must be positive, got {num_blocks}")
        if block_size <= 0:
            raise KVCacheError(f"block_size must be positive, got {block_size}")
        self.spec = spec
        self.num_blocks = num_blocks
        self.block_size = block_size
        shape = (num_blocks, block_size, spec.num_key_value_heads, spec.head_dim)
        self.keys = [
            torch.empty(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]
        self.values = [
            torch.empty(shape, dtype=spec.dtype, device=spec.device)
            for _ in range(spec.num_layers)
        ]

    def write(
        self,
        layer_index: int,
        block_table: torch.Tensor,
        position_ids: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter ``[batch, kv_heads, tokens, head_dim]`` states into blocks.

        ``block_table`` is ``[batch, blocks]`` and ``position_ids``
        ``[batch, tokens]``; row ``i`` of each addresses request ``i``.
        """

        table_indices = torch.div(position_ids, self.block_size, rounding_mode="floor")
        block_offsets = torch.remainder(position_ids, self.block_size)
        block_ids = block_table.gather(1, table_indices)
        with torch.no_grad():
            self.keys[layer_index].index_put_(
                (block_ids, block_offsets), keys.transpose(1, 2)
            )
            self.values[layer_index].index_put_(
                (block_ids, block_offsets), values.transpose(1, 2)
            )

    def gather(
        self, layer_index: int, block_table: torch.Tensor, length: int
    ) -> GatheredKV:
        """Collect one request's logical prefix as contiguous SDPA tensors."""

        keys = self.keys[layer_index].index_select(0, block_table)
        values = self.values[layer_index].index_select(0, block_table)
        # Page layout is [block, token, kv_head, dim]; SDPA consumes
        # [batch, kv_head, token, dim]. This gather is the reference backend.
        keys = keys.flatten(0, 1)[:length].permute(1, 0, 2).unsqueeze(0)
        values = values.flatten(0, 1)[:length].permute(1, 0, 2).unsqueeze(0)
        return GatheredKV(keys=keys, values=values)
