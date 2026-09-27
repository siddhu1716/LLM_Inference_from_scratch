"""Grouped-query causal self-attention with a simple KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .rope import RotaryEmbedding


@dataclass(frozen=True)
class KVCache:
    """Cached projected keys and values, each ``[batch, kv_heads, seq, head_dim]``."""

    keys: Tensor
    values: Tensor

    @property
    def seq_len(self) -> int:
        """Number of cached token positions."""
        return self.keys.size(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head or grouped-query self-attention.

    ``n_kv_heads == n_heads`` gives ordinary MHA.  Fewer KV heads gives GQA,
    reducing cache memory at the cost of sharing keys and values among query
    heads.  Inputs and outputs are ``[batch, seq, dim]``.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.repeats = n_heads // n_kv_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)

    def forward(
        self,
        x: Tensor,
        *,
        start_pos: int,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        batch, seq_len, _ = x.shape
        query = self._split_heads(self.q_proj(x), self.n_heads)
        key = self._split_heads(self.k_proj(x), self.n_kv_heads)
        value = self._split_heads(self.v_proj(x), self.n_kv_heads)
        query, key = self.rope(query, key, start_pos=start_pos)

        past_len = 0
        if cache is not None:
            self._validate_cache(cache, batch, x)
            past_len = cache.seq_len
            key = torch.cat((cache.keys, key), dim=-2)
            value = torch.cat((cache.values, value), dim=-2)
        new_cache = KVCache(keys=key, values=value)

        expanded_key = key.repeat_interleave(self.repeats, dim=1)
        expanded_value = value.repeat_interleave(self.repeats, dim=1)
        scores = query @ expanded_key.transpose(-2, -1)
        scores = scores * (self.head_dim**-0.5)

        query_positions = torch.arange(seq_len, device=x.device) + past_len
        key_positions = torch.arange(key.size(-2), device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        output = weights @ expanded_value
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output), new_cache

    def _split_heads(self, x: Tensor, heads: int) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def _validate_cache(self, cache: KVCache, batch: int, x: Tensor) -> None:
        expected = (batch, self.n_kv_heads)
        if cache.keys.shape[:2] != expected or cache.values.shape[:2] != expected:
            raise ValueError("cache batch or KV-head count does not match the input")
        if cache.keys.shape != cache.values.shape or cache.keys.size(-1) != self.head_dim:
            raise ValueError("cached keys and values must have matching valid shapes")
        if cache.keys.device != x.device or cache.keys.dtype != x.dtype:
            raise ValueError("cache device and dtype must match the model input")
