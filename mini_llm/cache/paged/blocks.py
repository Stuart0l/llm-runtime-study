"""Physical block bookkeeping for the paged KV cache."""

from __future__ import annotations

from typing import Iterable

from mini_llm.cache.contracts import KVCacheError


class BlockAllocator:
    """Fixed-size free list handing out exclusive physical block IDs."""

    __slots__ = ("num_blocks", "_free_blocks")

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise KVCacheError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        # Allocate low IDs first and deterministically reuse released blocks.
        self._free_blocks = list(reversed(range(num_blocks)))

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    def allocate(self, count: int) -> list[int]:
        if count > self.free_blocks:
            raise KVCacheError(
                "KV block pool exhausted: need "
                f"{count} blocks but only {self.free_blocks} are free"
            )
        return [self._free_blocks.pop() for _ in range(count)]

    def free(self, blocks: Iterable[int]) -> None:
        self._free_blocks.extend(blocks)
        self._free_blocks.sort(reverse=True)
