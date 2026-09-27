from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from math import exp, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory

from mini_infer.optimizations import (
    LoRAAdapter,
    NGramProposer,
    ParallelismConfig,
    PrefillDecodeRouter,
    StaticShapeGraphCache,
    StructuredDecoder,
    WorkerKind,
    dequantize_grouped_int4,
    dequantize_symmetric_int8,
    estimate_kv_cache,
    json_object_dfa,
    load_json_weight_shards,
    lora_linear,
    max_kv_cache_tokens,
    merge_row_shards,
    online_tiled_attention,
    plan_parallelism,
    quantize_grouped_int4,
    quantize_symmetric_int8,
    shard_rows,
    speculative_decode_step,
    top_k_route,
)


def assert_raises(exception_type: type[BaseException], callback: Callable[[], object]) -> None:
    try:
        callback()
    except exception_type:
        return
    raise AssertionError(f"{exception_type.__name__} was not raised")


def naive_attention(
    query: tuple[tuple[float, ...], ...],
    key: tuple[tuple[float, ...], ...],
    value: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    scale = 1 / sqrt(len(query[0]))
    output = []
    for query_row in query:
        scores = [
            sum(a * b for a, b in zip(query_row, key_row, strict=True)) * scale
            for key_row in key
        ]
        maximum = max(scores)
        probabilities = [exp(score - maximum) for score in scores]
        denominator = sum(probabilities)
        output.append(
            tuple(
                sum(
                    probability * row[index]
                    for probability, row in zip(probabilities, value, strict=True)
                )
                / denominator
                for index in range(len(value[0]))
            )
        )
    return tuple(output)


def test_kv_cache_estimator_and_capacity_inverse() -> None:
    estimate = estimate_kv_cache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        sequence_length=16,
        batch_size=3,
        bytes_per_element=2,
    )
    assert estimate.bytes_per_token_per_sequence == 2 * 2 * 4 * 8 * 2
    assert estimate.total_bytes == estimate.bytes_per_token_per_sequence * 16 * 3
    assert (
        max_kv_cache_tokens(
            estimate.total_bytes,
            num_layers=2,
            num_kv_heads=4,
            head_dim=8,
            batch_size=3,
            bytes_per_element=2,
        )
        == 16
    )


def test_online_tiled_attention_matches_materialized_softmax() -> None:
    query = ((1.0, 0.5), (-0.5, 2.0))
    key = ((0.5, 1.0), (2.0, -1.0), (0.25, 0.25))
    value = ((1.0, 2.0), (3.0, -1.0), (0.0, 4.0))
    actual = online_tiled_attention(query, key, value, tile_size=2)
    expected = naive_attention(query, key, value)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for actual_value, expected_value in zip(actual_row, expected_row, strict=True):
            assert abs(actual_value - expected_value) < 1e-12


def test_online_tiled_attention_causal_mask() -> None:
    output = online_tiled_attention(
        ((1.0,), (1.0,)),
        ((0.0,), (10.0,)),
        ((2.0,), (8.0,)),
        tile_size=1,
        causal=True,
    )
    assert output[0] == (2.0,)
    assert output[1][0] > 7.99


def test_symmetric_int8_and_grouped_int4_quantization_round_trip() -> None:
    weights = ((-1.0, 0.0, 1.0), (0.25, -0.5, 0.75))
    int8 = quantize_symmetric_int8(weights)
    restored8 = dequantize_symmetric_int8(int8)
    assert int8.shape == (2, 3)
    assert max(
        abs(original - restored)
        for row, restored_row in zip(weights, restored8, strict=True)
        for original, restored in zip(row, restored_row, strict=True)
    ) <= int8.scale / 2 + 1e-12

    values = (-1.0, -0.5, 0.0, 0.5, 1.0)
    int4 = quantize_grouped_int4(values, group_size=2)
    restored4 = dequantize_grouped_int4(int4)
    assert len(int4.packed_values) == 3
    assert len(int4.scales) == 3
    for index, (original, restored) in enumerate(zip(values, restored4, strict=True)):
        assert abs(original - restored) <= int4.scales[index // 2] / 2 + 1e-12


class SequenceRandom:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def test_speculative_decoding_rejects_and_samples_positive_residual() -> None:
    def draft(_prefix: tuple[int, ...]) -> Mapping[int, float]:
        return {1: 0.8, 2: 0.2}

    def target(_prefix: tuple[int, ...]) -> Mapping[int, float]:
        return {1: 0.4, 2: 0.6}

    result = speculative_decode_step(
        (),
        (1,),
        draft,
        target,
        rng=SequenceRandom([0.9, 0.1]),
    )
    assert result.rejected
    assert result.accepted_draft_tokens == 0
    assert result.token_ids == (2,)


def test_speculative_decoding_accepts_all_and_draws_bonus_token() -> None:
    def draft(_prefix: tuple[int, ...]) -> Mapping[int, float]:
        return {1: 1.0}

    def target(prefix: tuple[int, ...]) -> Mapping[int, float]:
        return {1: 1.0} if len(prefix) < 2 else {2: 1.0}

    result = speculative_decode_step(
        (),
        (1, 1),
        draft,
        target,
        rng=SequenceRandom([0.0, 0.0, 0.0]),
    )
    assert not result.rejected
    assert result.accepted_draft_tokens == 2
    assert result.token_ids == (1, 1, 2)


def test_ngram_proposer_reuses_latest_matching_context() -> None:
    proposer = NGramProposer(n=3)
    assert proposer.propose((7, 1, 2, 3, 4, 1, 2), max_tokens=2) == (3, 4)
    assert proposer.propose((1, 2), max_tokens=2) == ()


def test_lora_linear_applies_scaled_low_rank_update() -> None:
    adapter = LoRAAdapter(
        a=((1.0, 0.0),),
        b=((2.0,), (-1.0,)),
        alpha=2.0,
    )
    output = lora_linear(
        (3.0, 4.0),
        ((1.0, 0.0), (0.0, 1.0)),
        adapter,
        bias=(1.0, 1.0),
    )
    assert output == (16.0, -1.0)


def test_static_shape_graph_cache_simulates_capture_and_replay() -> None:
    cache: StaticShapeGraphCache[int] = StaticShapeGraphCache()
    captures = 0

    def capture() -> Callable[[int], int]:
        nonlocal captures
        captures += 1
        return lambda value: value * 2

    assert cache.execute((1, 8), capture, 3) == 6
    assert cache.execute((1, 8), capture, 4) == 8
    assert cache.execute((2, 8), capture, 5) == 10
    assert captures == cache.capture_count == 2
    assert cache.replay_count == 1
    assert cache.cached_shapes == ((1, 8), (2, 8))


def test_parallel_partition_plan_covers_all_axes() -> None:
    config = ParallelismConfig(tensor=2, pipeline=2, data=2, expert=2, context=2)
    plan = plan_parallelism(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        num_experts=4,
        config=config,
        sequence_length=12,
    )
    assert config.world_size == 32
    assert len(plan.ranks) == 32
    assert set(plan.ranks[0].layer_range) == {0, 1, 2, 3}
    assert plan.ranks[0].context_range is not None
    assert set(plan.ranks[0].context_range or ()) == set(range(6))
    assert {rank.pipeline_rank for rank in plan.ranks} == {0, 1}
    assert {rank.context_rank for rank in plan.ranks} == {0, 1}


def test_prefill_decode_disaggregation_routes_least_loaded_and_handoff() -> None:
    router = PrefillDecodeRouter(("p1", "p2"), ("d1", "d2"))
    first = router.route("r1", WorkerKind.PREFILL)
    second = router.route("r2", WorkerKind.PREFILL)
    decode = router.route("r1", WorkerKind.DECODE)
    assert (first.worker_id, second.worker_id) == ("p1", "p2")
    assert decode.worker_id == "d1"
    assert decode.transferred_from == "p1"
    router.complete("p1")
    assert router.load("p1") == 0


def test_moe_top_k_routing_is_normalized_and_deterministic_on_ties() -> None:
    routed = top_k_route(((2.0, 2.0, 1.0), (0.0, 3.0, 2.0)), k=2)
    assert [assignment.expert_index for assignment in routed[0]] == [0, 1]
    assert abs(sum(assignment.weight for assignment in routed[0]) - 1.0) < 1e-12
    assert [assignment.expert_index for assignment in routed[1]] == [1, 2]


def test_structured_decoder_enforces_json_like_token_dfa() -> None:
    decoder = StructuredDecoder(json_object_dfa())
    assert decoder.allowed_tokens == frozenset({"{"})
    for token in ("{", "STRING", ":", "NUMBER", "}"):
        decoder.consume(token)
    assert decoder.is_accepting
    assert decoder.allowed_tokens == frozenset()

    invalid = StructuredDecoder(json_object_dfa())
    invalid.consume("{")
    assert_raises(ValueError, lambda: invalid.consume(":"))


def test_weight_loading_and_row_sharding_round_trip() -> None:
    with TemporaryDirectory() as directory:
        first = Path(directory, "first.json")
        second = Path(directory, "second.json")
        first.write_text(json.dumps({"a": [[1, 2], [3, 4]]}), encoding="utf-8")
        second.write_text(json.dumps({"b": [[5, 6]]}), encoding="utf-8")
        loaded = load_json_weight_shards([first, second])
    assert loaded == {
        "a": ((1.0, 2.0), (3.0, 4.0)),
        "b": ((5.0, 6.0),),
    }
    matrix = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))
    shards = shard_rows(matrix, shard_count=2)
    assert shards == (matrix[:2], matrix[2:])
    assert merge_row_shards(shards) == matrix
