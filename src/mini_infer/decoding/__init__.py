"""Composable token-selection algorithms."""

from mini_infer.decoding.beam_search import Beam, beam_search
from mini_infer.decoding.categorical import (
    SeededCategoricalSampler,
    categorical_sample,
    sample_categorical,
)
from mini_infer.decoding.frequency_presence import apply_frequency_presence_penalties
from mini_infer.decoding.greedy import greedy_decode
from mini_infer.decoding.logit_bias import apply_logit_bias
from mini_infer.decoding.logits import argmax, log_softmax, softmax, validate_logits
from mini_infer.decoding.min_p import min_p_filter
from mini_infer.decoding.pipeline import Sampler, SamplingConfig
from mini_infer.decoding.repetition_penalty import apply_repetition_penalty
from mini_infer.decoding.stopping import (
    StopMatch,
    find_stop_match,
    matches_stop_sequence,
    matches_stop_token,
    should_stop,
)
from mini_infer.decoding.temperature import apply_temperature
from mini_infer.decoding.top_k import top_k_filter
from mini_infer.decoding.top_p import top_p_filter
from mini_infer.decoding.typical import typical_filter

__all__ = [
    "Beam",
    "Sampler",
    "SamplingConfig",
    "SeededCategoricalSampler",
    "StopMatch",
    "apply_frequency_presence_penalties",
    "apply_logit_bias",
    "apply_repetition_penalty",
    "apply_temperature",
    "argmax",
    "beam_search",
    "categorical_sample",
    "find_stop_match",
    "greedy_decode",
    "log_softmax",
    "matches_stop_sequence",
    "matches_stop_token",
    "min_p_filter",
    "sample_categorical",
    "should_stop",
    "softmax",
    "top_k_filter",
    "top_p_filter",
    "typical_filter",
    "validate_logits",
]
