"""Speculative decoding with exact rejection sampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol, TypeAlias

Distribution: TypeAlias = Mapping[int, float]
DistributionCallback: TypeAlias = Callable[[tuple[int, ...]], Distribution]


class RandomSource(Protocol):
    def random(self) -> float:
        ...


@dataclass(frozen=True, slots=True)
class SpeculativeStep:
    token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    rejected: bool


def _normalize(distribution: Distribution) -> dict[int, float]:
    if not distribution:
        raise ValueError("probability distribution must not be empty")
    if any(probability < 0 for probability in distribution.values()):
        raise ValueError("probabilities must be non-negative")
    total = sum(distribution.values())
    if total <= 0:
        raise ValueError("probability distribution must have positive mass")
    return {token_id: probability / total for token_id, probability in distribution.items()}


def _sample(distribution: Mapping[int, float], rng: RandomSource) -> int:
    threshold = rng.random()
    cumulative = 0.0
    ordered = sorted(distribution.items())
    for token_id, probability in ordered:
        cumulative += probability
        if threshold < cumulative:
            return token_id
    return ordered[-1][0]


def speculative_decode_step(
    prefix: Sequence[int],
    draft_tokens: Sequence[int],
    draft_distribution: DistributionCallback,
    target_distribution: DistributionCallback,
    *,
    rng: RandomSource | None = None,
) -> SpeculativeStep:
    """Verify draft tokens and emit a residual sample or one target bonus token."""
    if not draft_tokens:
        raise ValueError("draft_tokens must not be empty")
    random_source: RandomSource = Random() if rng is None else rng
    context = tuple(prefix)
    emitted: list[int] = []
    for accepted_count, draft_token in enumerate(draft_tokens):
        q = _normalize(draft_distribution(context))
        p = _normalize(target_distribution(context))
        q_token = q.get(draft_token, 0.0)
        if q_token <= 0:
            raise ValueError("draft token has zero probability under the draft distribution")
        acceptance_probability = min(1.0, p.get(draft_token, 0.0) / q_token)
        if random_source.random() < acceptance_probability:
            emitted.append(draft_token)
            context += (draft_token,)
            continue
        residual = {
            token_id: max(0.0, p.get(token_id, 0.0) - q.get(token_id, 0.0))
            for token_id in p.keys() | q.keys()
        }
        residual = _normalize(residual)
        emitted.append(_sample(residual, random_source))
        return SpeculativeStep(tuple(emitted), accepted_count, True)
    bonus = _sample(_normalize(target_distribution(context)), random_source)
    emitted.append(bonus)
    return SpeculativeStep(tuple(emitted), len(draft_tokens), False)
