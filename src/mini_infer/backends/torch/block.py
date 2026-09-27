"""A pre-normalized decoder Transformer block."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import CausalSelfAttention, KVCache
from .mlp import SwiGLU
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """Compose causal attention and SwiGLU with pre-norm residual paths."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        rope_theta: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.attention = CausalSelfAttention(dim, n_heads, n_kv_heads, rope_theta)
        self.mlp_norm = RMSNorm(dim, norm_eps)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(
        self,
        x: Tensor,
        *,
        start_pos: int,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        attention_output, new_cache = self.attention(
            self.attention_norm(x),
            start_pos=start_pos,
            cache=cache,
        )
        x = x + attention_output
        return x + self.mlp(self.mlp_norm(x)), new_cache
