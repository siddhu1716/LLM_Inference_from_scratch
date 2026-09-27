"""Small synchronous inference engine driven by a model callback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import monotonic
from typing import TypeAlias
from uuid import uuid4

from mini_infer.engine.batching import ContinuousBatcher
from mini_infer.engine.metrics import MetricsCollector
from mini_infer.engine.phases import ExecutionPhase
from mini_infer.engine.request import Request, RequestState
from mini_infer.engine.scheduler import TokenBudgetScheduler
from mini_infer.engine.streaming import EventStream, EventType, StreamEvent

ModelCallback: TypeAlias = Callable[[tuple[int, ...], ExecutionPhase], int]


class SynchronousEngine:
    def __init__(
        self,
        model: ModelCallback,
        *,
        max_batch_size: int = 4,
        max_batch_tokens: int = 2048,
        eos_token_id: int | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.model = model
        self.batcher = ContinuousBatcher(max_batch_size)
        self.scheduler = TokenBudgetScheduler(max_batch_tokens)
        self.events = EventStream()
        self.metrics = MetricsCollector()
        self.eos_token_id = eos_token_id
        self._clock = clock
        self._requests: dict[str, Request] = {}

    def submit(
        self,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        *,
        request_id: str | None = None,
    ) -> Request:
        tokens = tuple(prompt_tokens)
        if len(tokens) > self.scheduler.max_batch_tokens:
            raise ValueError("prompt exceeds the engine's per-step token budget")
        request = Request(request_id or uuid4().hex, tokens, max_new_tokens, self._clock())
        self.batcher.submit(request)
        self._requests[request.request_id] = request
        self.metrics.record_submitted(request.request_id, request.created_at)
        return request

    def get(self, request_id: str) -> Request:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request_id: {request_id!r}") from exc

    def step(self) -> tuple[Request, ...]:
        active = self.batcher.iteration()
        scheduled = self.scheduler.schedule(active)
        progressed: list[Request] = []
        for item in scheduled:
            request = item.request
            try:
                if request.state is RequestState.WAITING:
                    timestamp = self._clock()
                    request.transition(RequestState.PREFILLING, now=timestamp)
                    self.metrics.record_started(request.request_id, timestamp)
                    self.events.publish(StreamEvent(request.request_id, EventType.START))
                    token_id = self.model(request.prompt_tokens, ExecutionPhase.PREFILL)
                    request.transition(RequestState.DECODING, now=self._clock())
                else:
                    token_id = self.model(request.all_tokens, ExecutionPhase.DECODE)
                token_time = self._clock()
                request.append_token(token_id, now=token_time)
                self.metrics.record_token(request.request_id, token_time)
                self.events.publish(
                    StreamEvent(
                        request.request_id,
                        EventType.TOKEN,
                        token_id=token_id,
                        token_index=len(request.generated_tokens) - 1,
                    )
                )
                if token_id == self.eos_token_id or request.remaining_tokens == 0:
                    reason = "eos" if token_id == self.eos_token_id else "length"
                    finish_time = self._clock()
                    request.transition(RequestState.FINISHED, now=finish_time)
                    self.metrics.record_finished(request.request_id, finish_time)
                    self.events.publish(
                        StreamEvent(
                            request.request_id,
                            EventType.FINISH,
                            finish_reason=reason,
                        )
                    )
                progressed.append(request)
            except Exception as exc:
                failure_time = self._clock()
                request.fail(str(exc), now=failure_time)
                self.metrics.record_finished(request.request_id, failure_time)
                self.events.publish(
                    StreamEvent(request.request_id, EventType.ERROR, error=str(exc))
                )
                raise
        return tuple(progressed)

    def run_until_complete(self) -> dict[str, tuple[int, ...]]:
        while self.batcher.has_work:
            if not self.step() and self.batcher.has_work:
                raise RuntimeError("engine made no progress; check scheduling limits")
        return {
            request_id: tuple(request.generated_tokens)
            for request_id, request in self._requests.items()
        }
