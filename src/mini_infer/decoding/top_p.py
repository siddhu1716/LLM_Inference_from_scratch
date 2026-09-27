"""Nucleus (top-p) logit filtering."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import softmax, validate_min_tokens_to_keep


def top_p_filter(
    logits: list[float],
    top_p: float,
    *,
    min_tokens_to_keep: int = 1,
) -> list[float]:
    """Keep the smallest high-probability prefix whose mass reaches ``top_p``."""
    probabilities = softmax(logits)
    validate_min_tokens_to_keep(min_tokens_to_keep, len(logits))
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be a real number")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")

    ranked = sorted(range(len(logits)), key=lambda index: (-logits[index], index))
    kept: set[int] = set()
    cumulative_probability = 0.0
    for index in ranked:
        kept.add(index)
        cumulative_probability += probabilities[index]
        if len(kept) >= min_tokens_to_keep and cumulative_probability >= top_p:
            break
    return [float(value) if index in kept else -math.inf for index, value in enumerate(logits)]
