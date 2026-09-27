"""Mixture-of-Experts top-k routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp


@dataclass(frozen=True, slots=True)
class ExpertAssignment:
    token_index: int
    expert_index: int
    weight: float


def top_k_route(
    router_logits: Sequence[Sequence[float]],
    *,
    k: int,
) -> tuple[tuple[ExpertAssignment, ...], ...]:
    logits = tuple(tuple(float(value) for value in row) for row in router_logits)
    if not logits or not logits[0]:
        raise ValueError("router_logits must be a non-empty matrix")
    expert_count = len(logits[0])
    if any(len(row) != expert_count for row in logits):
        raise ValueError("router_logits must be rectangular")
    if not 1 <= k <= expert_count:
        raise ValueError("k must be between 1 and the expert count")
    routed: list[tuple[ExpertAssignment, ...]] = []
    for token_index, row in enumerate(logits):
        selected = sorted(range(expert_count), key=lambda index: (-row[index], index))[:k]
        maximum = max(row[index] for index in selected)
        unnormalized = [exp(row[index] - maximum) for index in selected]
        total = sum(unnormalized)
        routed.append(
            tuple(
                ExpertAssignment(token_index, expert_index, probability / total)
                for expert_index, probability in zip(
                    selected,
                    unnormalized,
                    strict=True,
                )
            )
        )
    return tuple(routed)
