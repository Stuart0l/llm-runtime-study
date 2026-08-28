"""Demonstrate how absolute positions rotate Qwen3 query/key features."""

from __future__ import annotations

import torch

from mini_llm.nn import (
    RotaryEmbedding,
    apply_rotary_position_embeddings,
    build_position_ids,
)


def main() -> int:
    rope = RotaryEmbedding(4, theta=10_000.0, max_position_embeddings=16)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])

    print("RoPE inverse frequencies:")
    print(f"  {rope.inverse_frequencies.tolist()}")
    print("Position-by-position rotation of [1, 2, 3, 4]:")
    for position in (0, 1, 5):
        position_ids = build_position_ids(1, offset=position)
        cosine, sine = rope(position_ids)
        queries = vector.reshape(1, 1, 1, 4)
        keys = vector.reshape(1, 1, 1, 4)
        rotated_queries, _ = apply_rotary_position_embeddings(
            queries, keys, cosine, sine
        )
        print(f"  position {position:>2}:")
        print(f"    cosine: {cosine[0, 0].tolist()}")
        print(f"    sine:   {sine[0, 0].tolist()}")
        print(f"    output: {rotated_queries[0, 0, 0].tolist()}")

    decode_positions = build_position_ids(3, offset=10)
    print("Cached decode position IDs for offset 10:")
    print(f"  {decode_positions.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
