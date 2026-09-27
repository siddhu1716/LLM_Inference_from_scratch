"""Small, dependency-free utilities for lists of logits."""

from __future__ import annotations

import math


def validate_logits(logits: list[float]) -> None:
    """Validate a non-empty logit vector.

    Negative infinity is allowed because filtering functions use it to mask
    tokens. NaN, positive infinity, and an entirely masked vector are invalid.
    """
    if not isinstance(logits, list):
        raise TypeError("logits must be a list")
    if not logits:
        raise ValueError("logits must not be empty")

    has_available_token = False
    for index, value in enumerate(logits):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"logits[{index}] must be a real number")
        numeric_value = float(value)
        if math.isnan(numeric_value) or numeric_value == math.inf:
            raise ValueError(f"logits[{index}] must not be NaN or positive infinity")
        if numeric_value != -math.inf:
            has_available_token = True

    if not has_available_token:
        raise ValueError("at least one logit must be finite")


def validate_min_tokens_to_keep(min_tokens_to_keep: int, vocabulary_size: int) -> None:
    if isinstance(min_tokens_to_keep, bool) or not isinstance(min_tokens_to_keep, int):
        raise TypeError("min_tokens_to_keep must be an integer")
    if not 1 <= min_tokens_to_keep <= vocabulary_size:
        raise ValueError("min_tokens_to_keep must be between 1 and the vocabulary size")


def argmax(logits: list[float]) -> int:
    """Return the first index containing the largest logit."""
    validate_logits(logits)
    return max(range(len(logits)), key=logits.__getitem__)


def softmax(logits: list[float]) -> list[float]:
    """Convert logits to probabilities using a stable softmax."""
    validate_logits(logits)
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) if value != -math.inf else 0.0 for value in logits]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def log_softmax(logits: list[float]) -> list[float]:
    """Convert logits to normalized log probabilities."""
    validate_logits(logits)
    maximum = max(logits)
    log_normalizer = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in logits if value != -math.inf)
    )
    return [
        value - log_normalizer if value != -math.inf else -math.inf
        for value in logits
    ]
