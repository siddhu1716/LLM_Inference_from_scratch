"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """Gated MLP mapping ``[..., dim]`` through ``hidden_dim`` and back."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
