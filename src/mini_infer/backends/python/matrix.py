"""Small nested-list linear algebra helpers used by the pure-Python backend."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

Vector = list[float]
Matrix = list[Vector]


def zeros(rows: int, columns: int) -> Matrix:
    """Return a ``[rows, columns]`` matrix with independent rows."""
    if rows < 0 or columns < 0:
        raise ValueError("matrix dimensions cannot be negative")
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def random_matrix(
    rows: int,
    columns: int,
    rng: random.Random,
    *,
    std: float,
) -> Matrix:
    """Initialize a ``[rows, columns]`` matrix from a local seeded RNG."""
    return [[rng.gauss(0.0, std) for _ in range(columns)] for _ in range(rows)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the dot product of equally sized vectors."""
    if len(left) != len(right):
        raise ValueError("dot-product vectors must have equal lengths")
    return sum(a * b for a, b in zip(left, right, strict=True))


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Swap a rectangular matrix's row and column axes."""
    if not matrix:
        return []
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    return [[matrix[row][column] for row in range(len(matrix))] for column in range(columns)]


def matmul(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> Matrix:
    """Multiply ``[m, k]`` and ``[k, n]`` matrices into ``[m, n]``."""
    if not left or not right:
        return []
    left_width = len(left[0])
    if any(len(row) != left_width for row in left):
        raise ValueError("left matrix must be rectangular")
    if any(len(row) != len(right[0]) for row in right):
        raise ValueError("right matrix must be rectangular")
    if left_width != len(right):
        raise ValueError("inner matrix dimensions must match")
    right_columns = transpose(right)
    return [[dot(row, column) for column in right_columns] for row in left]


def linear(vector: Sequence[float], weight: Sequence[Sequence[float]]) -> Vector:
    """Apply a bias-free linear layer with weight shape ``[out, in]``."""
    if weight and len(vector) != len(weight[0]):
        raise ValueError("input width does not match linear weight")
    if any(len(row) != len(vector) for row in weight):
        raise ValueError("linear weight must be rectangular")
    return [dot(row, vector) for row in weight]


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Add equally sized vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have equal lengths")
    return [a + b for a, b in zip(left, right, strict=True)]


def softmax(values: Sequence[float]) -> Vector:
    """Compute a numerically stable softmax over a one-dimensional row."""
    if not values:
        raise ValueError("softmax requires at least one value")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]
