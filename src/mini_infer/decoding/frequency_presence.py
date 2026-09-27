"""Frequency and presence penalties."""

from __future__ import annotations

import math
from collections import Counter

from mini_infer.decoding.logits import validate_logits
from mini_infer.decoding.repetition_penalty import _unique_valid_token_ids


def apply_frequency_presence_penalties(
    logits: list[float],
    token_ids: list[int],
    *,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> list[float]:
    """Subtract count-based frequency and one-off presence penalties."""
    validate_logits(logits)
    _validate_penalty(frequency_penalty, "frequency_penalty")
    _validate_penalty(presence_penalty, "presence_penalty")
    _unique_valid_token_ids(token_ids, len(logits))

    counts = Counter(token_ids)
    return [
        float(value)
        - frequency_penalty * counts.get(token_id, 0)
        - presence_penalty * (token_id in counts)
        for token_id, value in enumerate(logits)
    ]


def _validate_penalty(penalty: float, name: str) -> None:
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(penalty):
        raise ValueError(f"{name} must be finite")
