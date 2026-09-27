"""Locally typical sampling filter."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import softmax, validate_min_tokens_to_keep


def typical_filter(
    logits: list[float],
    typical_p: float,
    *,
    min_tokens_to_keep: int = 1,
) -> list[float]:
    """Keep tokens nearest the distribution's expected information content."""
    probabilities = softmax(logits)
    validate_min_tokens_to_keep(min_tokens_to_keep, len(logits))
    if isinstance(typical_p, bool) or not isinstance(typical_p, (int, float)):
        raise TypeError("typical_p must be a real number")
    if not math.isfinite(typical_p) or not 0 < typical_p <= 1:
        raise ValueError("typical_p must be finite and in (0, 1]")

    entropy = -math.fsum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )
    ranked = sorted(
        (index for index, probability in enumerate(probabilities) if probability > 0),
        key=lambda index: (
            abs(-math.log(probabilities[index]) - entropy),
            index,
        ),
    )
    kept: set[int] = set()
    cumulative_probability = 0.0
    for index in ranked:
        kept.add(index)
        cumulative_probability += probabilities[index]
        if len(kept) >= min_tokens_to_keep and cumulative_probability >= typical_p:
            break
    return [float(value) if index in kept else -math.inf for index, value in enumerate(logits)]
