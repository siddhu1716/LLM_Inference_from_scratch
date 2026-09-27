"""Min-p logit filtering."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import validate_logits, validate_min_tokens_to_keep


def min_p_filter(
    logits: list[float],
    min_p: float,
    *,
    min_tokens_to_keep: int = 1,
) -> list[float]:
    """Keep tokens whose probability is at least ``min_p`` times the maximum."""
    validate_logits(logits)
    validate_min_tokens_to_keep(min_tokens_to_keep, len(logits))
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)):
        raise TypeError("min_p must be a real number")
    if not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError("min_p must be finite and in [0, 1]")

    ranked = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    required = set(ranked[:min_tokens_to_keep])
    threshold = max(logits) + math.log(min_p) if min_p > 0 else -math.inf
    return [
        float(value) if index in required or value >= threshold else -math.inf
        for index, value in enumerate(logits)
    ]
