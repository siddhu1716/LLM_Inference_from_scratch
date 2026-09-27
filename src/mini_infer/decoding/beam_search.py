"""A small, model-agnostic beam search."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from mini_infer.decoding.logits import log_softmax


@dataclass(frozen=True, slots=True)
class Beam:
    """One beam-search hypothesis."""

    tokens: tuple[int, ...]
    score: float
    finished: bool = False

    @property
    def log_probability(self) -> float:
        return self.score


def beam_search(
    next_logits: Callable[[list[int]], list[float]],
    *,
    initial_tokens: list[int] | None = None,
    beam_width: int = 1,
    max_new_tokens: int = 1,
    eos_token_id: int | None = None,
    length_penalty: float = 0.0,
) -> list[Beam]:
    """Return the best hypotheses produced by repeatedly calling ``next_logits``."""
    if not callable(next_logits):
        raise TypeError("next_logits must be callable")
    prefix = [] if initial_tokens is None else initial_tokens
    _validate_token_ids(prefix, "initial_tokens")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        raise TypeError("beam_width must be an integer")
    if beam_width <= 0:
        raise ValueError("beam_width must be greater than zero")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if eos_token_id is not None:
        _validate_token_id(eos_token_id, "eos_token_id")
    if isinstance(length_penalty, bool) or not isinstance(length_penalty, (int, float)):
        raise TypeError("length_penalty must be a real number")
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("length_penalty must be finite and non-negative")

    prefix_length = len(prefix)
    beams = [Beam(tuple(prefix), 0.0, False)]
    for _ in range(max_new_tokens):
        candidates: list[Beam] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue
            logits = next_logits(list(beam.tokens))
            if not isinstance(logits, list):
                raise TypeError("next_logits must return a list of logits")
            if eos_token_id is not None and eos_token_id >= len(logits):
                raise ValueError("eos_token_id is outside the returned vocabulary")
            for token_id, token_log_probability in enumerate(log_softmax(logits)):
                if token_log_probability == -math.inf:
                    continue
                candidates.append(
                    Beam(
                        beam.tokens + (token_id,),
                        beam.score + token_log_probability,
                        token_id == eos_token_id,
                    )
                )
        beams = sorted(
            candidates,
            key=lambda beam: (
                -_rank_score(beam, prefix_length, float(length_penalty)),
                beam.tokens,
            ),
        )[:beam_width]
        if all(beam.finished for beam in beams):
            break
    return sorted(
        beams,
        key=lambda beam: (
            -_rank_score(beam, prefix_length, float(length_penalty)),
            beam.tokens,
        ),
    )


def _rank_score(beam: Beam, prefix_length: int, length_penalty: float) -> float:
    generated_length = max(1, len(beam.tokens) - prefix_length)
    return beam.score / generated_length**length_penalty


def _validate_token_ids(token_ids: list[int], name: str) -> None:
    if not isinstance(token_ids, list):
        raise TypeError(f"{name} must be a list")
    for index, token_id in enumerate(token_ids):
        _validate_token_id(token_id, f"{name}[{index}]")


def _validate_token_id(token_id: int, name: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{name} must be an integer")
    if token_id < 0:
        raise ValueError(f"{name} must be non-negative")
