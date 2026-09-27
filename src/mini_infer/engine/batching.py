"""Continuous batching with slot refill between decoding iterations."""

from __future__ import annotations

from collections import deque

from mini_infer.engine.request import Request


class ContinuousBatcher:
    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self._pending: deque[Request] = deque()
        self._active: dict[str, Request] = {}
        self._known_ids: set[str] = set()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active(self) -> tuple[Request, ...]:
        return tuple(self._active.values())

    def submit(self, request: Request) -> None:
        if request.request_id in self._known_ids:
            raise ValueError(f"duplicate request_id: {request.request_id!r}")
        self._known_ids.add(request.request_id)
        self._pending.append(request)

    def iteration(self) -> tuple[Request, ...]:
        for request_id, request in tuple(self._active.items()):
            if request.is_terminal:
                del self._active[request_id]
        while self._pending and len(self._active) < self.max_batch_size:
            request = self._pending.popleft()
            if not request.is_terminal:
                self._active[request.request_id] = request
        return self.active

    @property
    def has_work(self) -> bool:
        return bool(self._pending or self._active)
