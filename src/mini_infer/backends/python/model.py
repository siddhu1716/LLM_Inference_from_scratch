"""Dependency-free decoder-only Transformer model."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeAlias

from mini_infer.backends.python.attention import KVCache
from mini_infer.backends.python.block import TransformerBlock
from mini_infer.backends.python.embedding import Embedding
from mini_infer.backends.python.matrix import Matrix, linear, random_matrix
from mini_infer.backends.python.rms_norm import RMSNorm
from mini_infer.common.config import ModelConfig

ModelCache: TypeAlias = list[KVCache]


class Transformer:
    """A small decoder-only Transformer operating on one token sequence.

    Inputs have shape ``[sequence]`` and logits have shape
    ``[sequence, vocab]``. Use :meth:`forward_with_cache` during incremental
    decoding; each returned layer cache stores ``[tokens, kv_heads, head_dim]``
    keys and values.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        rng = random.Random(config.seed)
        self.embedding = Embedding(config.vocab_size, config.dim, rng)
        self.blocks = [TransformerBlock(config, rng) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output_weight = (
            self.embedding.weight
            if config.tie_embeddings
            else random_matrix(config.vocab_size, config.dim, rng, std=config.dim**-0.5)
        )

    def forward(
        self,
        input_ids: Sequence[int],
        cache: ModelCache | None = None,
        *,
        start_pos: int | None = None,
        use_cache: bool = False,
    ) -> Matrix | tuple[Matrix, ModelCache]:
        """Return logits, optionally paired with an updated KV cache."""
        logits, new_cache = self.forward_with_cache(input_ids, cache, start_pos=start_pos)
        return (logits, new_cache) if use_cache else logits

    def forward_with_cache(
        self,
        input_ids: Sequence[int],
        cache: ModelCache | None = None,
        *,
        start_pos: int | None = None,
    ) -> tuple[Matrix, ModelCache]:
        """Return ``[sequence, vocab]`` logits and a cache extended by these tokens."""
        token_ids = list(input_ids)
        if not token_ids:
            raise ValueError("input_ids cannot be empty")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must contain one entry per Transformer layer")
        cached_tokens = 0 if cache is None else len(cache[0])
        if cache is not None and any(len(layer_cache) != cached_tokens for layer_cache in cache):
            raise ValueError("all layer caches must contain the same number of tokens")
        position = cached_tokens if start_pos is None else start_pos
        if position != cached_tokens:
            raise ValueError("start_pos must equal the number of cached tokens")

        hidden = self.embedding(token_ids)
        new_cache: ModelCache = []
        for index, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[index]
            hidden, updated = block(hidden, layer_cache, start_pos=position)
            new_cache.append(updated)
        normalized = self.norm(hidden)
        logits = [linear(vector, self.output_weight) for vector in normalized]
        return logits, new_cache

    __call__ = forward
