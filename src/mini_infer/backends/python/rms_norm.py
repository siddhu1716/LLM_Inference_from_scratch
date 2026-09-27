"""Root-mean-square normalization without centering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mini_infer.backends.python.matrix import Matrix, Vector


class RMSNorm:
    """Normalize the last dimension and apply a learned ``[dim]`` scale.

    RMSNorm avoids the mean subtraction used by LayerNorm, making the operation
    especially easy to inspect in the dependency-free implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.weight = [1.0] * dim
        self.eps = eps

    def normalize(self, vector: Sequence[float]) -> Vector:
        """Normalize one ``[dim]`` vector."""
        if len(vector) != len(self.weight):
            raise ValueError("RMSNorm input width does not match its weight")
        inverse_rms = 1.0 / math.sqrt(
            sum(value * value for value in vector) / len(vector) + self.eps
        )
        return [
            value * inverse_rms * scale
            for value, scale in zip(vector, self.weight, strict=True)
        ]

    def __call__(self, inputs: Sequence[Sequence[float]]) -> Matrix:
        """Normalize a ``[sequence, dim]`` activation matrix."""
        return [self.normalize(vector) for vector in inputs]
