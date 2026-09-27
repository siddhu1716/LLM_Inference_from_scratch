"""LoRA linear adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _matrix(name: str, matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    result = tuple(tuple(float(value) for value in row) for row in matrix)
    if not result or not result[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    if any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be rectangular")
    return result


def _matvec(matrix: tuple[tuple[float, ...], ...], vector: tuple[float, ...]) -> list[float]:
    if len(matrix[0]) != len(vector):
        raise ValueError("matrix and vector dimensions do not match")
    return [
        sum(weight * value for weight, value in zip(row, vector, strict=True))
        for row in matrix
    ]


@dataclass(frozen=True, slots=True)
class LoRAAdapter:
    a: tuple[tuple[float, ...], ...]
    b: tuple[tuple[float, ...], ...]
    alpha: float = 1.0

    @property
    def rank(self) -> int:
        return len(self.a)


def lora_linear(
    inputs: Sequence[float],
    weights: Sequence[Sequence[float]],
    adapter: LoRAAdapter,
    bias: Sequence[float] | None = None,
) -> tuple[float, ...]:
    x = tuple(float(value) for value in inputs)
    w = _matrix("weights", weights)
    a = _matrix("adapter.a", adapter.a)
    b = _matrix("adapter.b", adapter.b)
    if len(a[0]) != len(x) or len(b) != len(w) or len(b[0]) != len(a):
        raise ValueError("LoRA adapter dimensions do not match the linear layer")
    base = _matvec(w, x)
    low_rank = _matvec(b, tuple(_matvec(a, x)))
    scale = adapter.alpha / len(a)
    if bias is not None:
        if len(bias) != len(base):
            raise ValueError("bias dimension does not match output dimension")
        base = [
            value + float(offset)
            for value, offset in zip(base, bias, strict=True)
        ]
    return tuple(
        value + scale * update
        for value, update in zip(base, low_rank, strict=True)
    )
