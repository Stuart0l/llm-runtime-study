"""Rotary position embeddings for decoder query and key heads."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    r"""Generate Qwen3/Llama-style rotary angles for absolute positions.

    The frequencies are fixed functions of ``head_dim`` and ``theta``; RoPE
    has no learned parameters.  They are registered as a non-persistent buffer
    so device movement follows the model without adding data to checkpoints.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        theta: float = 10_000.0,
        max_position_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_position_embeddings is not None and max_position_embeddings <= 0:
            raise ValueError(
                "max_position_embeddings must be positive when provided, got "
                f"{max_position_embeddings}"
            )

        self.head_dim = head_dim
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings
        inverse_frequencies = self._build_inverse_frequencies()
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def _build_inverse_frequencies(
        self, device: torch.device | str | None = None
    ) -> torch.Tensor:
        dimensions = torch.arange(
            0, self.head_dim, 2, dtype=torch.float32, device=device
        )
        return 1.0 / (self.theta ** (dimensions / self.head_dim))

    def materialize(self, device: torch.device | str = "cpu") -> None:
        """Create real values for a buffer produced during meta construction.

        When ``__init__`` runs under ``torch.device("meta")``, PyTorch executes
        ``_build_inverse_frequencies`` only to infer its output shape and dtype;
        no frequency values or storage exist.  Checkpoint loading cannot replace
        this non-persistent derived buffer, so the model loader calls this method
        once afterward to perform the actual numerical calculation.
        """

        self.inverse_frequencies = self._build_inverse_frequencies(device)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        output_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cosine and sine tables shaped ``[batch, sequence, head_dim]``."""

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.ndim != 2:
            raise ValueError(
                "position_ids must have shape [sequence] or [batch, sequence], got "
                f"{tuple(position_ids.shape)}"
            )
        if position_ids.dtype == torch.bool or position_ids.is_floating_point():
            raise TypeError(f"position_ids must use an integer dtype, got {position_ids.dtype}")
        if position_ids.numel() and torch.any(position_ids < 0).item():
            raise ValueError("position_ids must be non-negative")
        if (
            self.max_position_embeddings is not None
            and position_ids.numel()
            and torch.any(position_ids >= self.max_position_embeddings).item()
        ):
            maximum = int(position_ids.max().item())
            raise ValueError(
                f"position ID {maximum} exceeds the model limit "
                f"{self.max_position_embeddings - 1}"
            )
        if not output_dtype.is_floating_point:
            raise TypeError(f"output_dtype must be floating point, got {output_dtype}")

        positions = position_ids.to(
            device=self.inverse_frequencies.device, # type: ignore
            dtype=torch.float32,
        )
        frequencies = positions.unsqueeze(-1) * self.inverse_frequencies.view(1, 1, -1) # type: ignore
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(output_dtype), angles.sin().to(output_dtype)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )


def build_position_ids(
    sequence_length: int,
    *,
    offset: int = 0,
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build absolute positions for prompt prefill or offset cache decoding."""

    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    positions = torch.arange(
        offset,
        offset + sequence_length,
        dtype=torch.long,
        device=device,
    )
    return positions.unsqueeze(0).expand(batch_size, -1)


def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` to ``[-x2, x1]`` for Qwen3's half-split pairs."""

    first_half, second_half = inputs.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_position_embeddings(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K tensors shaped ``[batch, heads, sequence, head_dim]``."""

    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError(
            "RoPE expects Q/K shaped [batch, heads, sequence, head_dim]"
        )
    if queries.shape[0] != keys.shape[0]:
        raise ValueError("queries and keys must have the same batch size")
    if queries.shape[2] != keys.shape[2]:
        raise ValueError("queries and keys must have the same sequence length")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must have the same head dimension")
    expected_table_shape = (queries.shape[0], queries.shape[2], queries.shape[3])
    if cosine.shape != expected_table_shape or sine.shape != expected_table_shape:
        raise ValueError(
            "cosine and sine must both have shape [batch, sequence, head_dim] = "
            f"{expected_table_shape}, got {tuple(cosine.shape)} and {tuple(sine.shape)}"
        )

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_queries = queries * cosine + _rotate_half(queries) * sine
    rotated_keys = keys * cosine + _rotate_half(keys) * sine
    return rotated_queries, rotated_keys
