"""Cache-aware scheduling that prioritizes reusable prompt prefixes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from mini_infer.engine.prefix_cache import RadixPrefixCache
from mini_infer.engine.request import Request, RequestState

ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class CacheScheduleDecision:
    request: Request
    cached_tokens: int
    uncached_token_cost: int


class CacheAwareScheduler(Generic[ValueT]):
    def __init__(self, cache: RadixPrefixCache[ValueT], max_batch_tokens: int) -> None:
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        self.cache = cache
        self.max_batch_tokens = max_batch_tokens

    def schedule(self, requests: Iterable[Request]) -> tuple[CacheScheduleDecision, ...]:
        candidates: list[CacheScheduleDecision] = []
        for request in requests:
            if request.is_terminal:
                continue
            if request.state is RequestState.DECODING:
                candidates.append(CacheScheduleDecision(request, 0, 1))
                continue
            match = self.cache.longest_prefix(request.prompt_tokens)
            cached = 0 if match is None else match.length
            candidates.append(
                CacheScheduleDecision(request, cached, len(request.prompt_tokens) - cached)
            )
        candidates.sort(
            key=lambda decision: (
                -decision.cached_tokens,
                decision.request.created_at,
                decision.request.request_id,
            )
        )
        budget = self.max_batch_tokens
        selected: list[CacheScheduleDecision] = []
        for decision in candidates:
            if decision.uncached_token_cost <= budget:
                selected.append(decision)
                budget -= decision.uncached_token_cost
        return tuple(selected)
