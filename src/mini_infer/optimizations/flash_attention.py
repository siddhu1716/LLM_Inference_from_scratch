"""FlashAttention-style tiled exact attention using online softmax.

This pure-Python implementation demonstrates the numerically stable recurrence.
It is not a hardware kernel and is not expected to provide a speedup.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, sqrt


def _matrix(name: str, rows: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in rows)
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return matrix


def online_tiled_attention(
    query: Sequence[Sequence[float]],
    key: Sequence[Sequence[float]],
    value: Sequence[Sequence[float]],
    *,
    tile_size: int = 16,
    scale: float | None = None,
    causal: bool = False,
) -> tuple[tuple[float, ...], ...]:
    """Compute exact softmax(QK^T)V without materializing the score matrix."""
    q = _matrix("query", query)
    k = _matrix("key", key)
    v = _matrix("value", value)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if len(k) != len(v):
        raise ValueError("key and value must have the same row count")
    if len(q[0]) != len(k[0]):
        raise ValueError("query and key dimensions must match")
    if causal and len(q) != len(k):
        raise ValueError("causal attention requires equal query and key lengths")
    score_scale = 1.0 / sqrt(len(q[0])) if scale is None else scale
    output: list[tuple[float, ...]] = []
    for query_index, query_row in enumerate(q):
        running_max = float("-inf")
        running_sum = 0.0
        accumulator = [0.0] * len(v[0])
        for tile_start in range(0, len(k), tile_size):
            tile_end = min(tile_start + tile_size, len(k))
            for key_index in range(tile_start, tile_end):
                if causal and key_index > query_index:
                    continue
                score = (
                    sum(a * b for a, b in zip(query_row, k[key_index], strict=True))
                    * score_scale
                )
                new_max = max(running_max, score)
                old_factor = 0.0 if running_sum == 0.0 else exp(running_max - new_max)
                score_factor = exp(score - new_max)
                running_sum = running_sum * old_factor + score_factor
                for index, value_component in enumerate(v[key_index]):
                    accumulator[index] = (
                        accumulator[index] * old_factor + score_factor * value_component
                    )
                running_max = new_max
        if running_sum == 0.0:
            raise ValueError("attention row has no visible keys")
        output.append(tuple(component / running_sum for component in accumulator))
    return tuple(output)
