"""Trace tokens through a small top-k routed Granite MoE block."""

from __future__ import annotations

import torch

from mini_llm.nn import GraniteMoeBlock


def main() -> None:
    torch.manual_seed(23)
    num_experts = 5
    top_k = 2
    moe = GraniteMoeBlock(
        hidden_size=4,
        intermediate_size=3,
        num_experts=num_experts,
        top_k=top_k,
    )
    inputs = torch.randn(1, 3, 4)

    with torch.inference_mode():
        outputs, routing = moe(inputs)

    print("Granite sparse MoE trace")
    print(f"input shape:          {tuple(inputs.shape)}")
    print(f"output shape:         {tuple(outputs.shape)}")
    print(f"router logits dtype:  {routing.logits.dtype}")
    print(f"packed input weights: {tuple(moe.input_linear.weight.shape)}")
    print(f"packed output weights:{tuple(moe.output_linear.weight.shape)}")

    print("\nPer-token routing")
    for token_index, (indices, weights) in enumerate(
        zip(routing.expert_indices, routing.expert_weights, strict=True)
    ):
        selected = ", ".join(
            f"expert {int(index)}: {float(weight):.4f}"
            for index, weight in zip(indices, weights, strict=True)
        )
        print(f"token {token_index}: {selected} (sum={float(weights.sum()):.4f})")

    assignment_counts = torch.bincount(
        routing.expert_indices.flatten(), minlength=num_experts
    )
    print("\nExpert assignment counts")
    for expert_index, count in enumerate(assignment_counts.tolist()):
        print(f"expert {expert_index}: {count}")

    print(
        f"\nEach token used {top_k} of {num_experts} experts; "
        "unselected experts did no computation for that token."
    )


if __name__ == "__main__":
    main()
