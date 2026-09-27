"""Categorical sampling with explicit, seedable random state."""

from __future__ import annotations

import math
import random

from mini_infer.decoding.logits import softmax


def sample_categorical(probabilities: list[float], *, rng: random.Random) -> int:
    """Sample an index from non-negative, not-necessarily-normalized weights."""
    if not isinstance(probabilities, list):
        raise TypeError("probabilities must be a list")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be an instance of random.Random")

    total = 0.0
    for index, probability in enumerate(probabilities):
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise TypeError(f"probabilities[{index}] must be a real number")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"probabilities[{index}] must be finite and non-negative")
        total += float(probability)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("probabilities must have a finite positive sum")

    threshold = rng.random() * total
    cumulative = 0.0
    last_positive = 0
    for index, probability in enumerate(probabilities):
        if probability > 0:
            last_positive = index
        cumulative += probability
        if threshold < cumulative:
            return index
    return last_positive


def categorical_sample(
    logits: list[float],
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> int:
    """Sample from logits using either a seed or caller-owned random state."""
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    generator = rng if rng is not None else random.Random(seed)
    return sample_categorical(softmax(logits), rng=generator)


class SeededCategoricalSampler:
    """Stateful categorical sampler producing a reproducible sequence."""

    def __init__(self, seed: int | None = None) -> None:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        self._rng = random.Random(seed)

    def sample(self, logits: list[float]) -> int:
        return categorical_sample(logits, rng=self._rng)
