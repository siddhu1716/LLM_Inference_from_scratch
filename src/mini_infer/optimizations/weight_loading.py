"""Model weight loading from deterministic JSON shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

WeightMatrix: TypeAlias = tuple[tuple[float, ...], ...]


def load_json_weight_shards(paths: list[Path]) -> dict[str, WeightMatrix]:
    if not paths:
        raise ValueError("at least one weight shard path is required")
    loaded: dict[str, WeightMatrix] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"weight shard must contain an object: {path}")
        for name, raw_matrix in payload.items():
            if name in loaded:
                raise ValueError(f"duplicate tensor name across shards: {name!r}")
            if not isinstance(name, str) or not isinstance(raw_matrix, list) or not raw_matrix:
                raise ValueError(f"invalid tensor entry in shard: {path}")
            matrix = tuple(tuple(float(value) for value in row) for row in raw_matrix)
            if not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
                raise ValueError(f"tensor {name!r} must be a non-empty rectangular matrix")
            loaded[name] = matrix
    return loaded
