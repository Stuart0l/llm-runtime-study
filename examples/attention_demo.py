"""Trace shapes and causality through grouped-query self-attention."""

from __future__ import annotations

import torch

from mini_llm.nn import Qwen3Attention, RotaryEmbedding, build_position_ids


def main() -> None:
    torch.manual_seed(17)
    batch_size = 1
    sequence_length = 4
    hidden_size = 8
    query_heads = 4
    kv_heads = 2
    head_dim = 2

    attention = Qwen3Attention(
        hidden_size,
        query_heads,
        kv_heads,
        head_dim,
    ).eval()
    rope = RotaryEmbedding(head_dim)
    inputs = torch.randn(batch_size, sequence_length, hidden_size)
    cosine, sine = rope(build_position_ids(sequence_length))

    with torch.inference_mode():
        projected_queries = attention.q_proj(inputs)
        projected_keys = attention.k_proj(inputs)
        projected_values = attention.v_proj(inputs)
        outputs = attention(inputs, cosine, sine)

    print("Grouped-query causal attention trace")
    print(f"residual input:  {tuple(inputs.shape)}")
    print(f"query projection:{tuple(projected_queries.shape)}")
    print(f"key projection:  {tuple(projected_keys.shape)}")
    print(f"value projection:{tuple(projected_values.shape)}")
    print(
        "query heads:     "
        f"({batch_size}, {query_heads}, {sequence_length}, {head_dim})"
    )
    print(
        "KV heads:        "
        f"({batch_size}, {kv_heads}, {sequence_length}, {head_dim})"
    )
    print("KV sharing:      Q heads [0, 1] use KV 0; Q heads [2, 3] use KV 1")
    print(f"residual output: {tuple(outputs.shape)}")

    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.int32)
    )
    print("\nCausal visibility (rows=query token, columns=key token):")
    print(causal_mask)
    print("1 means visible; 0 means the future token is masked.")

    print("\nQwen3-0.6B uses the same flow at larger widths:")
    print("  residual: [B, T, 1024]")
    print("  Q:        [B, 16, T, 128]  (projection width 2048)")
    print("  K/V:      [B,  8, T, 128]  (projection width 1024)")
    print("  output:   [B, T, 1024]")


if __name__ == "__main__":
    main()
