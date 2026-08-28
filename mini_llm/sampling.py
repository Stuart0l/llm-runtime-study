"""Convert one vocabulary-logit vector into the next token ID."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


class SamplingError(ValueError):
    """Raised when sampling parameters or logits are invalid."""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Controls greedy or random next-token selection.

    ``temperature=0`` selects the largest logit deterministically. A positive
    temperature enables random sampling, optionally restricted by top-k and
    top-p filters.
    """

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise SamplingError("temperature must be finite and non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise SamplingError("top_k must be positive when supplied")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise SamplingError("top_p must satisfy 0 < top_p <= 1")
        if self.seed is not None and self.seed < 0:
            raise SamplingError("seed must be non-negative when supplied")


def filter_logits(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Apply temperature, top-k, and nucleus filters to one logits vector."""

    _validate_logits(logits)
    if config.temperature == 0:
        raise SamplingError("logit filtering requires a positive temperature")
    if config.top_k is not None and config.top_k > logits.numel():
        raise SamplingError(
            f"top_k {config.top_k} exceeds vocabulary size {logits.numel()}"
        )

    filtered = logits.float() / config.temperature
    if config.top_k is not None and config.top_k < filtered.numel():
        kept_logits, kept_indices = torch.topk(filtered, config.top_k)
        filtered = torch.full_like(filtered, -torch.inf).scatter(
            0, kept_indices, kept_logits
        )

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probability = torch.cumsum(sorted_probabilities, dim=-1)
        # Keep the first token that crosses top_p; otherwise a very small top_p
        # could remove every candidate.
        remove = cumulative_probability - sorted_probabilities >= config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        filtered = torch.empty_like(filtered).scatter(
            0, sorted_indices, sorted_logits
        )
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Return one token ID from a one-dimensional vocabulary-logit tensor."""

    _validate_logits(logits)
    if config.temperature == 0:
        return int(torch.argmax(logits).item())

    # Random sampling is deliberately performed in FP32 on CPU. This gives one
    # seeded generator implementation across CPU and Apple MPS; the model's
    # much larger forward computation remains on its selected device.
    cpu_logits = logits.detach().to(device="cpu", dtype=torch.float32)
    filtered = filter_logits(cpu_logits, config)
    probabilities = torch.softmax(filtered, dim=-1)
    token_id = torch.multinomial(probabilities, 1, generator=generator)
    return int(token_id.item())


def make_generator(seed: int | None) -> torch.Generator | None:
    """Create request-local random state without changing PyTorch's global seed."""

    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 1 or logits.numel() == 0:
        raise SamplingError(
            "logits must have shape [vocabulary] with at least one entry, got "
            f"{tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise SamplingError(f"logits must be floating point, got {logits.dtype}")
    if not torch.isfinite(logits).all().item():
        raise SamplingError("logits must contain only finite values")
