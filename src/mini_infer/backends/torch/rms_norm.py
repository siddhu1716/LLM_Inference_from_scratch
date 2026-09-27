"""Root-mean-square normalization."""

from __future__ import annotations

from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Normalize the last dimension without centering.

    Input and output have shape ``[..., dim]``.  The reduction is performed in
    float32 for stability, then converted back to the input dtype.  RMSNorm is
    cheaper than LayerNorm because it does not subtract the mean or learn a
    bias.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor(dim))
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * (x.float().square().mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return normalized.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)
