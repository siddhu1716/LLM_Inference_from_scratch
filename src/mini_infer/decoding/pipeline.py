"""Composable decoding pipeline."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from mini_infer.decoding.categorical import categorical_sample
from mini_infer.decoding.frequency_presence import apply_frequency_presence_penalties
from mini_infer.decoding.greedy import greedy_decode
from mini_infer.decoding.logit_bias import apply_logit_bias
from mini_infer.decoding.logits import validate_logits
from mini_infer.decoding.min_p import min_p_filter
from mini_infer.decoding.repetition_penalty import apply_repetition_penalty
from mini_infer.decoding.temperature import apply_temperature
from mini_infer.decoding.top_k import top_k_filter
from mini_infer.decoding.top_p import top_p_filter
from mini_infer.decoding.typical import typical_filter


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Configuration for penalty, filtering, and token-selection stages."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: Mapping[int, float] = field(default_factory=dict)
    greedy: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        _validate_real(self.temperature, "temperature", minimum=0.0)
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise TypeError("top_k must be an integer or None")
            if self.top_k <= 0:
                raise ValueError("top_k must be greater than zero")
        _validate_real(self.top_p, "top_p", minimum=0.0, maximum=1.0, exclusive_min=True)
        _validate_real(self.min_p, "min_p", minimum=0.0, maximum=1.0)
        _validate_real(
            self.typical_p,
            "typical_p",
            minimum=0.0,
            maximum=1.0,
            exclusive_min=True,
        )
        _validate_real(
            self.repetition_penalty,
            "repetition_penalty",
            minimum=0.0,
            exclusive_min=True,
        )
        _validate_real(self.frequency_penalty, "frequency_penalty")
        _validate_real(self.presence_penalty, "presence_penalty")
        if not isinstance(self.greedy, bool):
            raise TypeError("greedy must be a boolean")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if not isinstance(self.logit_bias, Mapping):
            raise TypeError("logit_bias must be a mapping")
        frozen_bias = dict(self.logit_bias)
        for token_id, amount in frozen_bias.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("logit_bias token ids must be integers")
            if token_id < 0:
                raise ValueError("logit_bias token ids must be non-negative")
            _validate_real(amount, f"logit_bias[{token_id}]")
        object.__setattr__(self, "logit_bias", MappingProxyType(frozen_bias))


class Sampler:
    """Apply a ``SamplingConfig`` to logits and select one token."""

    def __init__(self, config: SamplingConfig | None = None) -> None:
        self.config = SamplingConfig() if config is None else config
        if not isinstance(self.config, SamplingConfig):
            raise TypeError("config must be a SamplingConfig")
        self._rng = random.Random(self.config.seed)

    def prepare_logits(
        self,
        logits: list[float],
        token_ids: list[int] | None = None,
    ) -> list[float]:
        """Apply configured transformations without mutating caller-owned inputs."""
        validate_logits(logits)
        history = [] if token_ids is None else token_ids
        adjusted = apply_logit_bias(logits, self.config.logit_bias)
        adjusted = apply_repetition_penalty(
            adjusted,
            history,
            self.config.repetition_penalty,
        )
        adjusted = apply_frequency_presence_penalties(
            adjusted,
            history,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
        )
        if self.config.temperature > 0:
            adjusted = apply_temperature(adjusted, self.config.temperature)
        if self.config.top_k is not None:
            adjusted = top_k_filter(adjusted, self.config.top_k)
        if self.config.top_p < 1:
            adjusted = top_p_filter(adjusted, self.config.top_p)
        if self.config.min_p > 0:
            adjusted = min_p_filter(adjusted, self.config.min_p)
        if self.config.typical_p < 1:
            adjusted = typical_filter(adjusted, self.config.typical_p)
        return adjusted

    def sample(self, logits: list[float], token_ids: list[int] | None = None) -> int:
        """Select one token from transformed logits."""
        adjusted = self.prepare_logits(logits, token_ids)
        if self.config.greedy or self.config.temperature == 0:
            return greedy_decode(adjusted)
        return categorical_sample(adjusted, rng=self._rng)


def _validate_real(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if exclusive_min and value <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
        if not exclusive_min and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
