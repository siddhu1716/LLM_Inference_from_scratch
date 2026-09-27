"""SwiGLU feed-forward network."""

from __future__ import annotations

import math
import random

from mini_infer.backends.python.matrix import Matrix, linear, random_matrix
from mini_infer.common.config import ModelConfig


def silu(value: float) -> float:
    """Sigmoid linear unit used by SwiGLU."""
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


class SwiGLU:
    """Gated MLP mapping ``[sequence, dim]`` through ``ffn_dim`` and back."""

    def __init__(self, config: ModelConfig, rng: random.Random) -> None:
        std = config.dim**-0.5
        self.gate_proj = random_matrix(config.ffn_dim, config.dim, rng, std=std)
        self.up_proj = random_matrix(config.ffn_dim, config.dim, rng, std=std)
        self.down_proj = random_matrix(config.dim, config.ffn_dim, rng, std=std)

    def __call__(self, inputs: Matrix) -> Matrix:
        """Apply ``down(silu(gate(x)) * up(x))`` independently to each token."""
        output: Matrix = []
        for vector in inputs:
            gate = linear(vector, self.gate_proj)
            up = linear(vector, self.up_proj)
            hidden = [
                silu(gate_value) * up_value
                for gate_value, up_value in zip(gate, up, strict=True)
            ]
            output.append(linear(hidden, self.down_proj))
        return output
