from __future__ import annotations

import math
import random

import pytest

from mini_infer.decoding import (
    Sampler,
    SamplingConfig,
    SeededCategoricalSampler,
    apply_frequency_presence_penalties,
    apply_logit_bias,
    apply_repetition_penalty,
    apply_temperature,
    beam_search,
    categorical_sample,
    find_stop_match,
    greedy_decode,
    log_softmax,
    matches_stop_sequence,
    matches_stop_token,
    min_p_filter,
    sample_categorical,
    should_stop,
    softmax,
    top_k_filter,
    top_p_filter,
    typical_filter,
)


def finite_indices(values: list[float]) -> set[int]:
    return {index for index, value in enumerate(values) if math.isfinite(value)}


def test_softmax_and_log_softmax_are_stable_and_normalized() -> None:
    probabilities = softmax([1000.0, 1001.0, -math.inf])
    log_probabilities = log_softmax([1000.0, 1001.0, -math.inf])

    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] > probabilities[0] > probabilities[2]
    assert [math.exp(value) for value in log_probabilities[:2]] == pytest.approx(
        probabilities[:2]
    )
    assert log_probabilities[2] == -math.inf


@pytest.mark.parametrize("logits", [[], [math.nan], [math.inf], [-math.inf]])
def test_logit_utilities_reject_invalid_vectors(logits: list[float]) -> None:
    with pytest.raises(ValueError):
        softmax(logits)


def test_greedy_decode_uses_lowest_index_to_break_ties() -> None:
    assert greedy_decode([1.0, 3.0, 3.0]) == 1


def test_temperature_scales_without_mutating_input() -> None:
    logits = [2.0, -4.0]
    assert apply_temperature(logits, 2.0) == [1.0, -2.0]
    assert logits == [2.0, -4.0]
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)


def test_top_k_filter_keeps_exactly_the_best_k_with_deterministic_ties() -> None:
    assert finite_indices(top_k_filter([3.0, 2.0, 2.0, 1.0], 2)) == {0, 1}
    with pytest.raises(ValueError):
        top_k_filter([1.0], 2)


def test_top_p_filter_keeps_smallest_ranked_probability_prefix() -> None:
    filtered = top_p_filter([math.log(0.6), math.log(0.3), math.log(0.1)], 0.7)
    assert finite_indices(filtered) == {0, 1}
    assert top_p_filter([1.0, 0.0], 1.0) == [1.0, 0.0]


def test_min_p_filter_uses_probability_relative_to_most_likely_token() -> None:
    filtered = min_p_filter([0.0, math.log(0.5), math.log(0.1)], 0.2)
    assert finite_indices(filtered) == {0, 1}
    assert min_p_filter([0.0, -1.0], 0.0) == [0.0, -1.0]


def test_typical_filter_ranks_tokens_by_distance_from_entropy() -> None:
    logits = [math.log(0.5), math.log(0.3), math.log(0.2)]
    assert finite_indices(typical_filter(logits, 0.2)) == {1}
    assert typical_filter(logits, 1.0) == pytest.approx(logits)


def test_repetition_penalty_is_sign_aware_and_applied_once_per_token() -> None:
    logits = [4.0, -2.0, 1.0]
    adjusted = apply_repetition_penalty(logits, [0, 0, 1], 2.0)
    assert adjusted == [2.0, -4.0, 1.0]
    assert logits == [4.0, -2.0, 1.0]


def test_frequency_and_presence_penalties_use_counts() -> None:
    adjusted = apply_frequency_presence_penalties(
        [5.0, 5.0, 5.0],
        [0, 0, 1],
        frequency_penalty=0.5,
        presence_penalty=1.0,
    )
    assert adjusted == [3.0, 3.5, 5.0]


def test_logit_bias_is_additive_and_validates_vocabulary() -> None:
    assert apply_logit_bias([0.0, 1.0], {0: 2.5}) == [2.5, 1.0]
    assert apply_logit_bias([0.0, -math.inf], {1: 2.5}) == [0.0, -math.inf]
    with pytest.raises(ValueError):
        apply_logit_bias([0.0], {1: 1.0})


def test_seeded_categorical_sampling_is_reproducible() -> None:
    first = SeededCategoricalSampler(11)
    second = SeededCategoricalSampler(11)
    assert [first.sample([0.0, 1.0, 2.0]) for _ in range(12)] == [
        second.sample([0.0, 1.0, 2.0]) for _ in range(12)
    ]
    assert categorical_sample([0.0, 1.0], seed=7) == categorical_sample(
        [0.0, 1.0], seed=7
    )


def test_categorical_sampling_accepts_unnormalized_weights() -> None:
    assert sample_categorical([0.0, 10.0, 0.0], rng=random.Random(3)) == 1
    with pytest.raises(ValueError):
        sample_categorical([0.0, 0.0], rng=random.Random(3))


def test_stop_token_and_sequence_matching() -> None:
    assert matches_stop_token(2, [1, 2])
    assert matches_stop_sequence([4, 5, 6], [5, 6])
    assert should_stop([1, 2, 3], stop_sequences=[[2, 3]])
    assert find_stop_match(
        [1, 2, 3],
        stop_token_ids=[3],
        stop_sequences=[[3], [2, 3]],
    ).token_ids == (2, 3)
    assert not should_stop([1, 2], stop_token_ids=[3], stop_sequences=[[2, 3]])


def test_stop_matching_rejects_empty_configured_sequence() -> None:
    with pytest.raises(ValueError):
        should_stop([1], stop_sequences=[[]])


def test_beam_search_tracks_scores_eos_and_deterministic_order() -> None:
    def next_logits(tokens: list[int]) -> list[float]:
        if tokens and tokens[-1] == 0:
            return [-math.inf, -math.inf, 3.0]
        return [2.0, 1.0, 0.0]

    beams = beam_search(
        next_logits,
        beam_width=2,
        max_new_tokens=2,
        eos_token_id=2,
    )

    assert beams[0].tokens == (0, 2)
    assert beams[0].finished
    assert len(beams) == 2
    assert beams[0].score >= beams[1].score


def test_beam_search_validates_callback_output_and_arguments() -> None:
    with pytest.raises(TypeError):
        beam_search(lambda _: (1.0, 2.0))  # type: ignore[return-value]
    with pytest.raises(ValueError):
        beam_search(lambda _: [1.0], beam_width=0)


def test_sampling_config_is_validated_and_copies_logit_bias() -> None:
    bias = {1: 2.0}
    config = SamplingConfig(logit_bias=bias, seed=5)
    bias[1] = 99.0
    assert config.logit_bias[1] == 2.0
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0)


def test_sampler_pipeline_applies_transformations_and_is_seeded() -> None:
    config = SamplingConfig(
        temperature=0.7,
        top_k=3,
        top_p=0.9,
        min_p=0.05,
        typical_p=0.95,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
        presence_penalty=0.1,
        logit_bias={2: 0.5},
        seed=123,
    )
    first = Sampler(config)
    second = Sampler(config)
    logits = [0.0, 0.5, 1.0, 1.5]
    history = [2, 2]

    assert first.prepare_logits(logits, history) == second.prepare_logits(logits, history)
    assert [first.sample(logits, history) for _ in range(10)] == [
        second.sample(logits, history) for _ in range(10)
    ]
    assert logits == [0.0, 0.5, 1.0, 1.5]


def test_sampler_supports_explicit_and_zero_temperature_greedy_modes() -> None:
    assert Sampler(SamplingConfig(greedy=True, seed=1)).sample([0.0, 2.0, 1.0]) == 1
    assert Sampler(SamplingConfig(temperature=0.0, seed=1)).sample([0.0, 2.0, 1.0]) == 1
