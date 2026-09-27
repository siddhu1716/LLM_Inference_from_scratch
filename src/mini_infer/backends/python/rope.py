"""Rotary position embedding (RoPE) for attention queries and keys."""

from __future__ import annotations

import math
from collections.abc import Sequence

from mini_infer.backends.python.matrix import Vector


def apply_rope(vector: Sequence[float], position: int, theta: float = 10_000.0) -> Vector:
    """Rotate adjacent feature pairs in one ``[head_dim]`` vector.

    RoPE stores no position table here; computing sinusoids on demand is slower
    but keeps the educational implementation compact and transparent.
    """
    if len(vector) % 2:
        raise ValueError("RoPE requires an even head dimension")
    if position < 0:
        raise ValueError("position cannot be negative")
    output = [0.0] * len(vector)
    for pair_start in range(0, len(vector), 2):
        frequency = theta ** (-pair_start / len(vector))
        angle = position * frequency
        cosine = math.cos(angle)
        sine = math.sin(angle)
        first, second = vector[pair_start], vector[pair_start + 1]
        output[pair_start] = first * cosine - second * sine
        output[pair_start + 1] = first * sine + second * cosine
    return output
