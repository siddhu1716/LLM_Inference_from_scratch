"""Token embedding lookup for nested-list activations."""

from __future__ import annotations

import random

from mini_infer.backends.python.matrix import Matrix, random_matrix


class Embedding:
    """Store a ``[vocab, dim]`` table and return one row per token."""

    def __init__(self, vocab_size: int, dim: int, rng: random.Random) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = random_matrix(vocab_size, dim, rng, std=dim**-0.5)

    def __call__(self, token_ids: list[int]) -> Matrix:
        """Look up token IDs, producing ``[sequence, dim]`` activations."""
        output: Matrix = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token IDs must be integers")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            output.append(self.weight[token_id].copy())
        return output
