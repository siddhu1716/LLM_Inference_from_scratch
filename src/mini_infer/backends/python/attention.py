"""Causal multi-head and grouped-query self-attention."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mini_infer.backends.python.matrix import (
    Matrix,
    Vector,
    linear,
    random_matrix,
    softmax,
    zeros,
)
from mini_infer.backends.python.rope import apply_rope
from mini_infer.common.config import ModelConfig

Heads = list[Vector]


@dataclass(slots=True)
class KVCache:
    """Per-layer key/value history with shape ``[tokens, kv_heads, head_dim]``."""

    keys: list[Heads] = field(default_factory=list)
    values: list[Heads] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)


def _split_heads(vector: Vector, heads: int, head_dim: int) -> Heads:
    return [vector[index * head_dim : (index + 1) * head_dim] for index in range(heads)]


class CausalSelfAttention:
    """Reference causal attention supporting MHA and GQA.

    Query activations use ``n_heads`` while keys and values use ``kv_heads``.
    GQA shares each KV head across a contiguous group of query heads, reducing
    cache size from ``n_heads`` to ``kv_heads`` entries per token.
    """

    def __init__(self, config: ModelConfig, rng: random.Random) -> None:
        self.config = config
        std = config.dim**-0.5
        kv_width = config.kv_heads * config.head_dim
        self.q_proj = random_matrix(config.dim, config.dim, rng, std=std)
        self.k_proj = random_matrix(kv_width, config.dim, rng, std=std)
        self.v_proj = random_matrix(kv_width, config.dim, rng, std=std)
        self.o_proj = random_matrix(config.dim, config.dim, rng, std=std)

    def __call__(
        self,
        inputs: Matrix,
        cache: KVCache | None = None,
        *,
        start_pos: int = 0,
    ) -> tuple[Matrix, KVCache]:
        """Attend over ``[sequence, dim]`` inputs and return outputs plus new cache."""
        if start_pos < 0:
            raise ValueError("start_pos cannot be negative")
        previous = KVCache() if cache is None else cache
        if len(previous.keys) != len(previous.values):
            raise ValueError("cache key/value lengths differ")
        if len(previous) != start_pos:
            raise ValueError("start_pos must equal the number of cached tokens")
        if start_pos + len(inputs) > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")

        keys = [[head.copy() for head in token] for token in previous.keys]
        values = [[head.copy() for head in token] for token in previous.values]
        queries: list[Heads] = []
        for offset, vector in enumerate(inputs):
            position = start_pos + offset
            query_heads = _split_heads(
                linear(vector, self.q_proj), self.config.n_heads, self.config.head_dim
            )
            key_heads = _split_heads(
                linear(vector, self.k_proj), self.config.kv_heads, self.config.head_dim
            )
            value_heads = _split_heads(
                linear(vector, self.v_proj), self.config.kv_heads, self.config.head_dim
            )
            queries.append(
                [apply_rope(head, position, self.config.rope_theta) for head in query_heads]
            )
            keys.append(
                [apply_rope(head, position, self.config.rope_theta) for head in key_heads]
            )
            values.append(value_heads)

        output = zeros(len(inputs), self.config.dim)
        queries_per_kv = self.config.n_heads // self.config.kv_heads
        scale = self.config.head_dim**-0.5
        for token_offset, query_heads in enumerate(queries):
            last_visible = start_pos + token_offset
            concatenated: Vector = []
            for query_index, query in enumerate(query_heads):
                kv_index = query_index // queries_per_kv
                scores = [
                    sum(q * k for q, k in zip(query, keys[position][kv_index], strict=True))
                    * scale
                    for position in range(last_visible + 1)
                ]
                probabilities = softmax(scores)
                attended = [
                    sum(
                        probability * values[position][kv_index][feature]
                        for position, probability in enumerate(probabilities)
                    )
                    for feature in range(self.config.head_dim)
                ]
                concatenated.extend(attended)
            output[token_offset] = linear(concatenated, self.o_proj)
        return output, KVCache(keys, values)
