"""Sign-aware repetition penalty."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import validate_logits


def apply_repetition_penalty(
    logits: list[float],
    token_ids: list[int],
    penalty: float,
) -> list[float]:
    """Penalize each previously seen token once using the common sign-aware rule."""
    validate_logits(logits)
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
        raise TypeError("penalty must be a real number")
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and greater than zero")

    adjusted = [float(value) for value in logits]
    for token_id in _unique_valid_token_ids(token_ids, len(logits)):
        value = adjusted[token_id]
        adjusted[token_id] = value * penalty if value < 0 else value / penalty
    return adjusted


def _unique_valid_token_ids(token_ids: list[int], vocabulary_size: int) -> set[int]:
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list")
    unique: set[int] = set()
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"token_ids[{index}] must be an integer")
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(f"token_ids[{index}] is outside the vocabulary")
        unique.add(token_id)
    return unique
