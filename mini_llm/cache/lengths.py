"""Logical sequence length bookkeeping shared by every cache backend."""

from __future__ import annotations

from mini_llm.cache.contracts import KVCacheError


class SequenceLength:
    """How many positions of a fixed-capacity cache currently hold valid K/V.

    The count is a host-side Python int. Attention derives its bounds from
    ``position_ids``, which already lives on the device, so no step of the
    forward pass reads this value back off a tensor.
    """

    __slots__ = ("capacity", "length")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise KVCacheError(f"cache capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.length = 0

    def extend(self, token_count: int) -> None:
        """Commit ``token_count`` further positions, or raise if they do not fit."""

        if token_count <= 0:
            raise KVCacheError(f"token_count must be positive, got {token_count}")
        required = self.length + token_count
        if required > self.capacity:
            raise KVCacheError(
                f"KV cache capacity exceeded: need {required} positions but capacity "
                f"is {self.capacity}"
            )
        self.length = required

    def rollback(self, length: int) -> None:
        if length < 0 or length > self.length:
            raise KVCacheError(f"cannot roll cache back to length {length}")
        self.length = length

    def reset(self) -> None:
        self.length = 0
