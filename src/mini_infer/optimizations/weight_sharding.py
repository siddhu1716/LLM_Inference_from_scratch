"""Row-wise model-weight sharding."""

from __future__ import annotations

from collections.abc import Sequence


def shard_rows(
    matrix: Sequence[Sequence[float]],
    *,
    shard_count: int,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be non-empty and rectangular")
    if shard_count <= 0 or len(rows) % shard_count:
        raise ValueError("row count must be divisible by a positive shard_count")
    rows_per_shard = len(rows) // shard_count
    return tuple(
        rows[index : index + rows_per_shard]
        for index in range(0, len(rows), rows_per_shard)
    )


def merge_row_shards(
    shards: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[float, ...], ...]:
    if not shards:
        raise ValueError("shards must not be empty")
    merged = tuple(
        tuple(float(value) for value in row)
        for shard in shards
        for row in shard
    )
    if not merged or not merged[0] or any(len(row) != len(merged[0]) for row in merged):
        raise ValueError("shards must contain compatible rectangular rows")
    return merged
