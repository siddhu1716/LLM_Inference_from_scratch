"""In-process streaming events."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    START = "start"
    TOKEN = "token"
    FINISH = "finish"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    request_id: str
    event_type: EventType
    token_id: int | None = None
    token_index: int | None = None
    finish_reason: str | None = None
    error: str | None = None


class EventStream:
    def __init__(self) -> None:
        self._events: dict[str, deque[StreamEvent]] = defaultdict(deque)

    def publish(self, event: StreamEvent) -> None:
        if not event.request_id:
            raise ValueError("event request_id must not be empty")
        self._events[event.request_id].append(event)

    def drain(self, request_id: str) -> tuple[StreamEvent, ...]:
        queue = self._events.get(request_id)
        if queue is None:
            return ()
        events = tuple(queue)
        queue.clear()
        return events

    def pending(self, request_id: str) -> int:
        return len(self._events.get(request_id, ()))
