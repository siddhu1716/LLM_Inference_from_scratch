"""One pre-normalized decoder Transformer block."""

from __future__ import annotations

import random

from mini_infer.backends.python.attention import CausalSelfAttention, KVCache
from mini_infer.backends.python.matrix import Matrix, add
from mini_infer.backends.python.mlp import SwiGLU
from mini_infer.backends.python.rms_norm import RMSNorm
from mini_infer.common.config import ModelConfig


class TransformerBlock:
    """Compose RMSNorm, causal attention, SwiGLU, and residual connections."""

    def __init__(self, config: ModelConfig, rng: random.Random) -> None:
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention = CausalSelfAttention(config, rng)
        self.mlp_norm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = SwiGLU(config, rng)

    def __call__(
        self,
        inputs: Matrix,
        cache: KVCache | None = None,
        *,
        start_pos: int = 0,
    ) -> tuple[Matrix, KVCache]:
        """Transform ``[sequence, dim]`` activations and update one layer cache."""
        attended, new_cache = self.attention(
            self.attention_norm(inputs), cache, start_pos=start_pos
        )
        residual = [add(vector, update) for vector, update in zip(inputs, attended, strict=True)]
        mlp_output = self.mlp(self.mlp_norm(residual))
        return (
            [
                add(vector, update)
                for vector, update in zip(residual, mlp_output, strict=True)
            ],
            new_cache,
        )
