"""Demonstrate RMSNorm and Qwen3 per-head Q/K normalization."""

from __future__ import annotations

import torch

from mini_llm.nn import RMSNorm, normalize_qwen3_queries_and_keys


def main() -> int:
    inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    scale = torch.tensor([1.0, 0.5, 1.5, 2.0])
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(scale)

    squares = inputs.square()
    mean_square = squares.mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + norm.eps)
    unit_rms = inputs * inverse_rms
    output = norm(inputs)

    print("RMSNorm walkthrough:")
    print(f"  input:                 {inputs.tolist()}")
    print(f"  x squared:             {squares.tolist()}")
    print(f"  mean(x squared):       {mean_square.tolist()}")
    print(f"  reciprocal RMS:        {inverse_rms.tolist()}")
    print(f"  normalized x:          {unit_rms.tolist()}")
    print(f"  learned scale:         {scale.tolist()}")
    print(f"  final output:          {output.tolist()}")

    queries = inputs.reshape(1, 1, 1, 4).repeat(1, 2, 1, 1)
    keys = inputs.reshape(1, 1, 1, 4)
    query_norm = RMSNorm(4, eps=1e-6)
    key_norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        query_norm.weight.fill_(2.0)
        key_norm.weight.fill_(3.0)
    normalized_queries, normalized_keys = normalize_qwen3_queries_and_keys(
        queries,
        keys,
        query_norm=query_norm,
        key_norm=key_norm,
    )
    print("Qwen3 per-head Q/K normalization:")
    print(f"  query shape:           {tuple(queries.shape)}")
    print(f"  key shape:             {tuple(keys.shape)}")
    print(f"  first normalized Q:    {normalized_queries[0, 0, 0].tolist()}")
    print(f"  first normalized K:    {normalized_keys[0, 0, 0].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
