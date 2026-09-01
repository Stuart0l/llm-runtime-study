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

        # Routing is a discrete decision: a small FP16 kernel difference can
        # exchange the experts immediately above and below the top-k boundary,
        # after which CPU and MPS execute different networks. Compute the
        # projection itself in FP32, rather than projecting in FP16 and only
        # widening the already-rounded result. Expert projections still use the
        # model execution dtype.
        logits = F.linear(inputs.float(), self.layer.weight.float())
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

        if inputs.device.type in ("mps", "cuda"):
            if flattened.shape[0] > 1:
                return self._forward_multiple_tokens(inputs, flattened, routing)
            return self._forward_single_token(inputs, flattened, routing)

        combined = torch.zeros_like(flattened)

        # CPU keeps the readable active-expert loop: its low dispatch overhead
        # is preferable to gathering or widening large packed weight tensors.
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

    def _forward_single_token(
        self,
        inputs: torch.Tensor,
        flattened: torch.Tensor,
        routing: RoutingDecision,
    ) -> tuple[torch.Tensor, RoutingDecision]:
        """Evaluate selected accelerator decode experts with batched matmuls.

        Gathering eight packed matrices costs some memory bandwidth, but avoids
        converting expert IDs to a Python list and launching two independent
        matrix multiplications for every selected expert. Sorting preserves the
        expert accumulation order used by the readable multi-token path.
        """

        expert_indices, selection_order = routing.expert_indices[0].sort()
        expert_weights = routing.expert_weights[0, selection_order]

        input_weights = self.input_linear.weight.index_select(0, expert_indices)
        repeated_input = flattened.expand(self.top_k, -1).unsqueeze(-1)
        gate_and_up = torch.bmm(input_weights, repeated_input).squeeze(-1)
        gate, up = gate_and_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        output_weights = self.output_linear.weight.index_select(0, expert_indices)
        expert_outputs = torch.bmm(
            output_weights, hidden.unsqueeze(-1)
        ).squeeze(-1)
        combined = (
            expert_outputs.float() * expert_weights.unsqueeze(-1)
        ).sum(dim=0, keepdim=True)

        return combined.to(inputs.dtype).view_as(inputs), routing

    def _forward_multiple_tokens(
        self,
        inputs: torch.Tensor,
        flattened: torch.Tensor,
        routing: RoutingDecision,
    ) -> tuple[torch.Tensor, RoutingDecision]:
        """Run accelerator prefill as two padded expert-major batched matmuls.

        Every token creates ``top_k`` assignments in token-major order. A prefix
        count maps each assignment to a unique ``[expert, position]`` coordinate
        in a padded expert batch. Those same coordinates recover the results in
        their original order, avoiding sorting, inverse permutations, and
        scatter-add.
        """

        num_tokens = flattened.shape[0]
        num_assignments = num_tokens * self.top_k

        # Flattening preserves token-major order:
        #   token 0's K experts, token 1's K experts, ...
        # Keeping this order is what later makes a simple [tokens, K, hidden]
        # reshape sufficient to reassemble each token's expert results.
        expert_indices = routing.expert_indices.reshape(num_assignments)
        expert_weights = routing.expert_weights.reshape(num_assignments)

        # Calculate a zero-based position within each expert without sorting.
        # For expert IDs [2, 0, 2, 0], the positions are [0, 0, 1, 1], giving
        # unique coordinates (2, 0), (0, 0), (2, 1), and (0, 1).
        expert_markers = F.one_hot(
            expert_indices, num_classes=self.num_experts
        )
        expert_counts = expert_markers.sum(dim=0)
        max_assignments = int(expert_counts.max().item())
        cumulative_counts = expert_markers.cumsum(dim=0)
        positions = (
            cumulative_counts.gather(1, expert_indices.unsqueeze(1)).squeeze(1)
            - 1
        )

        # Repeat each token once for every selected expert, then place each
        # assignment at its [expert, position] coordinate. Unused cells remain
        # zero; Granite's bias-free expert projections keep them zero.
        assignment_inputs = (
            flattened.unsqueeze(1)
            .expand(num_tokens, self.top_k, self.hidden_size)
            .reshape(num_assignments, self.hidden_size)
        )
        expert_inputs = torch.zeros(
            self.num_experts,
            max_assignments,
            self.hidden_size,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        expert_inputs[expert_indices, positions] = assignment_inputs

        # The expert dimension is the batch dimension of both matrix
        # multiplications. Keep the larger gate/up projection in the model dtype.
        # Widen only the smaller output projection: FP16 output GEMM differences
        # were sufficient to change later routes, while FP16 gate/up remained
        # aligned in the CPU/MPS regression.
        gate_and_up = torch.bmm(
            expert_inputs,
            self.input_linear.weight.transpose(1, 2),
        )
        gate, up = gate_and_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        expert_outputs = torch.bmm(
            hidden.float(),
            self.output_linear.weight.float().transpose(1, 2),
        )

        # Read valid results through the same coordinates used for packing.
        # They are therefore already token-major: reshape to [tokens, K, hidden],
        # apply the router probabilities, and sum each token's K contributions.
        assignment_outputs = expert_outputs[expert_indices, positions]
        combined = (
            assignment_outputs.float() * expert_weights.unsqueeze(-1)
        ).view(num_tokens, self.top_k, self.hidden_size).sum(dim=1)
        return combined.to(inputs.dtype).view_as(inputs), routing
