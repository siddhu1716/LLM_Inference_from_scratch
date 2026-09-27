"""Request lifecycle and state transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    RequestState.WAITING: {
        RequestState.PREFILLING,
        RequestState.CANCELLED,
        RequestState.FAILED,
    },
    RequestState.PREFILLING: {
        RequestState.DECODING,
        RequestState.CANCELLED,
        RequestState.FAILED,
    },
    RequestState.DECODING: {
        RequestState.FINISHED,
        RequestState.CANCELLED,
        RequestState.FAILED,
    },
}


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    created_at: float = field(default_factory=monotonic)
    generated_tokens: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    started_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.FAILED,
        }

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - len(self.generated_tokens))

    @property
    def all_tokens(self) -> tuple[int, ...]:
        return self.prompt_tokens + tuple(self.generated_tokens)

    def transition(self, new_state: RequestState, *, now: float | None = None) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(f"invalid request transition: {self.state.value} -> {new_state.value}")
        timestamp = monotonic() if now is None else now
        self.state = new_state
        if new_state is RequestState.PREFILLING:
            self.started_at = timestamp
        if new_state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.FAILED,
        }:
            self.finished_at = timestamp

    def append_token(self, token_id: int, *, now: float | None = None) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError("tokens can only be appended while decoding")
        if self.remaining_tokens == 0:
            raise RuntimeError("request token limit has already been reached")
        self.generated_tokens.append(token_id)
        if self.first_token_at is None:
            self.first_token_at = monotonic() if now is None else now

    def cancel(self, *, now: float | None = None) -> None:
        if self.is_terminal:
            raise RuntimeError("terminal requests cannot be cancelled")
        self.transition(RequestState.CANCELLED, now=now)

    def fail(self, message: str, *, now: float | None = None) -> None:
        if not message:
            raise ValueError("failure message must not be empty")
        if self.is_terminal:
            raise RuntimeError("terminal requests cannot fail")
        self.error = message
        self.transition(RequestState.FAILED, now=now)
