"""Granite sparse MoE routing and packed SwiGLU experts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """FP32 router results for every flattened input token."""

    logits: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor


class TopKRouter(nn.Module):
    """Select and normalize the highest-scoring experts for each token."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(
                f"top_k must be within [1, {num_experts}], got {top_k}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        # The nested name ``router.layer.weight`` matches Granite's checkpoint.
        self.layer = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, inputs: torch.Tensor) -> RoutingDecision:
        if not inputs.is_floating_point():
            raise TypeError(f"router input must be floating point, got {inputs.dtype}")
        if inputs.ndim != 2 or inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                f"router expected [tokens, {self.hidden_size}], got "
                f"{tuple(inputs.shape)}"
            )

        # Match Granite's reference path: projection uses the model execution
        # dtype, while expert selection and softmax operate on FP32 logits.
        logits = self.layer(inputs).float()
        top_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        expert_weights = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
        return RoutingDecision(logits, expert_indices, expert_weights)


class PackedExpertLinear(nn.Module):
    """One rank-3 weight tensor containing the same projection for all experts."""

    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        super().__init__()
        for name, value in (
            ("num_experts", num_experts),
            ("input_size", input_size),
            ("output_size", output_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(
            torch.empty(num_experts, output_size, input_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert_weight in self.weight:
            nn.init.kaiming_uniform_(expert_weight, a=math.sqrt(5))

    def forward_expert(
        self, inputs: torch.Tensor, expert_index: int
    ) -> torch.Tensor:
        return F.linear(inputs, self.weight[expert_index])


class GraniteMoeBlock(nn.Module):
    """Top-k routed packed SwiGLU experts with token-wise scatter-add."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.input_linear = PackedExpertLinear(
            num_experts, hidden_size, 2 * intermediate_size
        )
        self.output_linear = PackedExpertLinear(
            num_experts, intermediate_size, hidden_size
        )
        self.router = TopKRouter(hidden_size, num_experts, top_k)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, RoutingDecision]:
        if not inputs.is_floating_point():
            raise TypeError(
                f"GraniteMoeBlock requires floating-point input, got {inputs.dtype}"
            )
        if inputs.ndim != 3 or inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                "GraniteMoeBlock expected [batch, sequence, "
                f"{self.hidden_size}], got {tuple(inputs.shape)}"
            )

        batch_size, sequence_length, _ = inputs.shape
        flattened = inputs.reshape(-1, self.hidden_size)
        routing = self.router(flattened)
        combined = torch.zeros_like(flattened)

        # Only experts selected by at least one token execute. Converting this
        # short list to Python makes the dispatch easy to inspect; its device
        # synchronization cost will be measured in the performance component.
        active_experts = torch.unique(routing.expert_indices, sorted=True).tolist()
        for expert_index in active_experts:
            token_indices, selected_slots = torch.where(
                routing.expert_indices == expert_index
            )
            expert_inputs = flattened[token_indices]
            gate_and_up = self.input_linear.forward_expert(
                expert_inputs, expert_index
            )
            gate, up = gate_and_up.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            expert_outputs = self.output_linear.forward_expert(
                hidden, expert_index
            )
            weights = routing.expert_weights[
                token_indices, selected_slots
            ].to(expert_outputs.dtype)
            combined = combined.index_add(
                0, token_indices, expert_outputs * weights.unsqueeze(-1)
            )

        return (
            combined.view(batch_size, sequence_length, self.hidden_size),
            routing,
        )
