"""Prefill-versus-decode phase classification."""

from __future__ import annotations

from enum import Enum

from mini_infer.engine.request import Request, RequestState


class ExecutionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def phase_for(request: Request) -> ExecutionPhase:
    if request.state in {RequestState.WAITING, RequestState.PREFILLING}:
        return ExecutionPhase.PREFILL
    if request.state is RequestState.DECODING:
        return ExecutionPhase.DECODE
    raise ValueError(f"terminal request {request.request_id!r} has no execution phase")


def phase_token_cost(request: Request) -> int:
    """Return tokens processed by an unchunked forward pass in the current phase."""
    return len(request.prompt_tokens) if phase_for(request) is ExecutionPhase.PREFILL else 1
