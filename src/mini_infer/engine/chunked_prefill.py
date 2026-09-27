"""Chunked-prefill planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillChunk:
    start: int
    end: int
    token_ids: tuple[int, ...]
    is_last: bool


class ChunkedPrefillPlanner:
    def __init__(self, chunk_size: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def chunks(self, prompt_tokens: Sequence[int]) -> tuple[PrefillChunk, ...]:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        result: list[PrefillChunk] = []
        for start in range(0, len(prompt_tokens), self.chunk_size):
            end = min(start + self.chunk_size, len(prompt_tokens))
            result.append(
                PrefillChunk(
                    start=start,
                    end=end,
                    token_ids=tuple(prompt_tokens[start:end]),
                    is_last=end == len(prompt_tokens),
                )
            )
        return tuple(result)

    def next_chunk(
        self,
        prompt_tokens: Sequence[int],
        cursor: int,
        token_budget: int | None = None,
    ) -> PrefillChunk | None:
        if not 0 <= cursor <= len(prompt_tokens):
            raise ValueError("cursor is outside prompt_tokens")
        if cursor == len(prompt_tokens):
            return None
        available = self.chunk_size if token_budget is None else min(self.chunk_size, token_budget)
        if available <= 0:
            raise ValueError("token_budget must be positive")
        end = min(cursor + available, len(prompt_tokens))
        return PrefillChunk(
            cursor,
            end,
            tuple(prompt_tokens[cursor:end]),
            end == len(prompt_tokens),
        )
