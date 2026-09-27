"""Top-k logit filtering."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import validate_logits, validate_min_tokens_to_keep


def top_k_filter(
    logits: list[float],
    k: int,
    *,
    min_tokens_to_keep: int = 1,
) -> list[float]:
    """Mask every token outside the ``k`` highest logits."""
    validate_logits(logits)
    validate_min_tokens_to_keep(min_tokens_to_keep, len(logits))
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 1 <= k <= len(logits):
        raise ValueError("k must be between 1 and the vocabulary size")

    keep_count = max(k, min_tokens_to_keep)
    kept_indices = sorted(range(len(logits)), key=lambda index: (-logits[index], index))[
        :keep_count
    ]
    kept = set(kept_indices)
    return [float(value) if index in kept else -math.inf for index, value in enumerate(logits)]
