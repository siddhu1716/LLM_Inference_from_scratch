"""Weight-only per-tensor symmetric int8 quantization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SymmetricInt8Weights:
    values: tuple[tuple[int, ...], ...]
    scale: float

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.values), len(self.values[0])


def _weights(weights: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in weights)
    if not matrix or not matrix[0]:
        raise ValueError("weights must be a non-empty matrix")
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("weights must be rectangular")
    return matrix


def quantize_symmetric_int8(weights: Sequence[Sequence[float]]) -> SymmetricInt8Weights:
    matrix = _weights(weights)
    maximum = max(abs(value) for row in matrix for value in row)
    scale = maximum / 127.0 if maximum else 1.0
    quantized = tuple(
        tuple(max(-127, min(127, round(value / scale))) for value in row)
        for row in matrix
    )
    return SymmetricInt8Weights(quantized, scale)


def dequantize_symmetric_int8(
    weights: SymmetricInt8Weights,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(value * weights.scale for value in row)
        for row in weights.values
    )
