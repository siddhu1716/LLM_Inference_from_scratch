"""Temperature scaling."""

from __future__ import annotations

import math

from mini_infer.decoding.logits import validate_logits


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    """Return logits divided by a strictly positive temperature."""
    validate_logits(logits)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    return [float(value) / float(temperature) for value in logits]
