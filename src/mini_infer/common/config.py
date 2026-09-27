"""Validated configuration shared by the educational Transformer backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture and initialization settings for a decoder-only Transformer.

    ``dim`` is the residual width. Each query head has width
    ``dim // n_heads``; ``n_kv_heads`` may be smaller than ``n_heads`` for
    grouped-query attention. Defaults intentionally describe a tiny model that
    is practical in the pure-Python backend.
    """

    vocab_size: int = 256
    dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int | None = None
    hidden_dim: int | None = None
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    seed: int = 0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Reject incompatible dimensions before either backend allocates weights."""
        positive_ints = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_kv_heads is not None and self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("dim // n_heads must be even for RoPE")
        if self.n_heads % self.kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.rope_theta <= 0.0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        """Width of one attention head."""
        return self.dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Resolved number of key/value heads (equal to query heads for MHA)."""
        return self.n_heads if self.n_kv_heads is None else self.n_kv_heads

    @property
    def ffn_dim(self) -> int:
        """Resolved SwiGLU hidden width, rounded to a small hardware-friendly multiple."""
        if self.hidden_dim is not None:
            return self.hidden_dim
        estimated = int(8 * self.dim / 3)
        return max(8, ((estimated + 7) // 8) * 8)
