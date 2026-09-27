"""Token-budget scheduling."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from mini_infer.engine.phases import ExecutionPhase, phase_for, phase_token_cost
from mini_infer.engine.request import Request


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    request: Request
    phase: ExecutionPhase
    token_cost: int


class TokenBudgetScheduler:
    def __init__(self, max_batch_tokens: int) -> None:
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        self.max_batch_tokens = max_batch_tokens

    def schedule(self, requests: Iterable[Request]) -> tuple[ScheduledRequest, ...]:
        budget = self.max_batch_tokens
        scheduled: list[ScheduledRequest] = []
        candidates = sorted(
            (request for request in requests if not request.is_terminal),
            key=lambda request: (request.created_at, request.request_id),
        )
        for request in candidates:
            cost = phase_token_cost(request)
            if cost > budget:
                continue
            scheduled.append(ScheduledRequest(request, phase_for(request), cost))
            budget -= cost
        return tuple(scheduled)
