"""Trace a small tensor through Qwen3's SwiGLU feed-forward network."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mini_llm.nn import SwiGLUFeedForward


def main() -> None:
    torch.manual_seed(7)
    hidden_size = 4
    intermediate_size = 8
    mlp = SwiGLUFeedForward(hidden_size, intermediate_size)
    inputs = torch.tensor([[[0.5, -1.0, 1.5, 2.0]]])

    with torch.inference_mode():
        gate_values = mlp.gate_proj(inputs)
        activated_gate = F.silu(gate_values)
        up_values = mlp.up_proj(inputs)
        gated_values = activated_gate * up_values
        outputs = mlp.down_proj(gated_values)
        module_outputs = mlp(inputs)

    print("Qwen3 SwiGLU trace")
    print(f"input          {tuple(inputs.shape)}: {inputs}")
    print(f"gate_proj      {tuple(gate_values.shape)}: {gate_values}")
    print(f"silu(gate)     {tuple(activated_gate.shape)}: {activated_gate}")
    print(f"up_proj        {tuple(up_values.shape)}: {up_values}")
    print(f"gated product  {tuple(gated_values.shape)}: {gated_values}")
    print(f"down_proj      {tuple(outputs.shape)}: {outputs}")
    torch.testing.assert_close(module_outputs, outputs)
    print("\nModule output matches the explicitly traced operations.")


if __name__ == "__main__":
    main()
