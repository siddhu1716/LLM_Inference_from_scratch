"""Small educational decoder-only Transformer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor, nn

from .attention import KVCache
from .block import TransformerBlock
from .rms_norm import RMSNorm


class ModelConfig(Protocol):
    """Structural type for the shared model configuration."""

    vocab_size: int
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int | None
    hidden_dim: int | None
    ffn_dim: int
    max_seq_len: int
    rope_theta: float
    norm_eps: float
    seed: int
    tie_embeddings: bool


LayerCache = KVCache | None
TransformerCache = list[KVCache]


class Transformer(nn.Module):
    """A compact decoder-only Transformer accepting ``[batch, seq]`` token IDs.

    The result is logits with shape ``[batch, seq, vocab_size]``.  ``forward``
    may consume a read-only per-layer cache but returns only logits;
    ``forward_with_cache`` also returns the extended cache needed for
    incremental decoding.  The cache is intentionally a transparent list of
    tensors rather than a production paged-cache implementation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        n_kv_heads = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        hidden_dim = config.ffn_dim
        _validate_config(config, n_kv_heads, hidden_dim)
        self.config = config
        self.max_seq_len = config.max_seq_len

        # fork_rng restores the caller's CPU RNG state after deterministic construction.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
            self.layers = nn.ModuleList(
                TransformerBlock(
                    config.dim,
                    config.n_heads,
                    n_kv_heads,
                    hidden_dim,
                    config.rope_theta,
                    config.norm_eps,
                )
                for _ in range(config.n_layers)
            )
            self.norm = RMSNorm(config.dim, config.norm_eps)
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
            self.apply(self._init_weights)

        if config.tie_embeddings:
            self.output.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        start_pos: int | None = None,
    ) -> Tensor:
        """Return logits, optionally attending to an existing per-layer cache."""
        logits, _ = self.forward_with_cache(input_ids, cache=cache, start_pos=start_pos)
        return logits

    def forward_with_cache(
        self,
        input_ids: Tensor,
        cache: Sequence[LayerCache] | None = None,
        start_pos: int | None = None,
    ) -> tuple[Tensor, TransformerCache]:
        """Return logits and an extended KV cache for prompt or token decoding.

        On the first call pass a prompt ``[B, T]`` and no cache.  Subsequent
        calls may pass ``[B, 1]`` and the returned cache.  Chunked continuation
        also works.  Cache growth uses ``torch.cat`` for clarity; production
        inference preallocates or pages cache storage to avoid copying.
        """
        _validate_input(input_ids)
        layer_cache = self._normalize_cache(cache)
        inferred_pos = 0 if layer_cache[0] is None else layer_cache[0].seq_len
        position = inferred_pos if start_pos is None else start_pos
        if position < 0:
            raise ValueError("start_pos must be non-negative")
        if position != inferred_pos:
            raise ValueError("start_pos must equal the number of cached positions")
        if position + input_ids.size(1) > self.max_seq_len:
            raise ValueError("sequence exceeds config.max_seq_len")

        x = self.token_embeddings(input_ids)
        new_cache: TransformerCache = []
        for layer, cached in zip(self.layers, layer_cache, strict=True):
            x, updated = layer(x, start_pos=position, cache=cached)
            new_cache.append(updated)
        return self.output(self.norm(x)), new_cache

    def _normalize_cache(
        self,
        cache: Sequence[LayerCache] | None,
    ) -> list[LayerCache]:
        if cache is None:
            return [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError(f"cache must contain {len(self.layers)} layer entries")
        lengths = {entry.seq_len for entry in cache if entry is not None}
        if len(lengths) > 1:
            raise ValueError("all layer caches must have the same sequence length")
        if any(entry is None for entry in cache) and any(entry is not None for entry in cache):
            raise ValueError("cache entries must be either all populated or all empty")
        return list(cache)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _validate_config(config: ModelConfig, n_kv_heads: int, hidden_dim: int) -> None:
    positive = {
        "vocab_size": config.vocab_size,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_dim": hidden_dim,
        "max_seq_len": config.max_seq_len,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"configuration values must be positive: {', '.join(invalid)}")
    if config.dim % config.n_heads:
        raise ValueError("dim must be divisible by n_heads")
    if config.n_heads % n_kv_heads:
        raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
    if (config.dim // config.n_heads) % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    if config.rope_theta <= 0 or config.norm_eps <= 0:
        raise ValueError("rope_theta and norm_eps must be positive")


def _validate_input(input_ids: Tensor) -> None:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    if input_ids.size(0) == 0 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have non-empty batch and sequence dimensions")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must contain integer token indices")
