"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Apply RoPE to query and key heads.

    Query and key tensors are ``[batch, heads, seq, head_dim]``.  RoPE rotates
    adjacent feature pairs and has no learned parameters.  Computing angles on
    demand keeps this implementation small; production systems usually cache
    sine and cosine tables.
    """

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        *,
        start_pos: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Rotate ``query`` and ``key`` at positions beginning at ``start_pos``."""
        seq_len = query.size(-2)
        if key.size(-2) != seq_len:
            raise ValueError("query and key must cover the same positions before caching")
        positions = torch.arange(
            start_pos,
            start_pos + seq_len,
            device=query.device,
            dtype=self.inv_freq.dtype,
        )
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos()[None, None, :, :].to(dtype=query.dtype)
        sin = angles.sin()[None, None, :, :].to(dtype=query.dtype)
        return _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    pairs = x.unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(dim=-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)
