"""Deterministic n-gram speculative proposal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NGramProposer:
    n: int = 3

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("n must be at least 2")

    def propose(self, token_ids: Sequence[int], max_tokens: int) -> tuple[int, ...]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tokens = tuple(token_ids)
        context_size = self.n - 1
        if len(tokens) < self.n:
            return ()
        context = tokens[-context_size:]
        latest_start: int | None = None
        for start in range(0, len(tokens) - context_size):
            if tokens[start : start + context_size] == context:
                latest_start = start
        if latest_start is None:
            return ()
        continuation_start = latest_start + context_size
        return tokens[continuation_start : continuation_start + max_tokens]
