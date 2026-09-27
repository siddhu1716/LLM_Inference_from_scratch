"""Per-token additive logit bias."""

from __future__ import annotations

import math
from collections.abc import Mapping

from mini_infer.decoding.logits import validate_logits


def apply_logit_bias(logits: list[float], bias: Mapping[int, float]) -> list[float]:
    """Add configured bias values without mutating the input logits."""
    validate_logits(logits)
    if not isinstance(bias, Mapping):
        raise TypeError("bias must be a mapping from token ids to real numbers")

    adjusted = [float(value) for value in logits]
    for token_id, amount in bias.items():
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("logit bias token ids must be integers")
        if not 0 <= token_id < len(logits):
            raise ValueError(f"logit bias token id {token_id} is outside the vocabulary")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"logit bias for token {token_id} must be a real number")
        if not math.isfinite(amount):
            raise ValueError(f"logit bias for token {token_id} must be finite")
        adjusted[token_id] += float(amount)
        if math.isnan(adjusted[token_id]) or adjusted[token_id] == math.inf:
            raise ValueError(f"logit bias for token {token_id} produced a non-finite logit")
    return adjusted
