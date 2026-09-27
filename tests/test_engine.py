from __future__ import annotations

from collections.abc import Callable

from mini_infer.engine import (
    BlockAllocator,
    CacheAwareScheduler,
    ChunkedPrefillPlanner,
    ContinuousBatcher,
    EventType,
    ExecutionPhase,
    MetricsCollector,
    PagedKVCache,
    RadixPrefixCache,
    Request,
    RequestState,
    SynchronousEngine,
    TokenBudgetScheduler,
)


def assert_raises(exception_type: type[BaseException], callback: Callable[[], object]) -> None:
    try:
        callback()
    except exception_type:
        return
    raise AssertionError(f"{exception_type.__name__} was not raised")


def decoding_request(request_id: str, created_at: float = 0.0) -> Request:
    request = Request(request_id, (1, 2), 3, created_at=created_at)
    request.transition(RequestState.PREFILLING, now=created_at + 0.1)
    request.transition(RequestState.DECODING, now=created_at + 0.2)
    return request


def test_request_lifecycle_enforces_state_and_token_limit() -> None:
    request = Request("r", (1, 2), 1, created_at=10.0)
    assert request.state is RequestState.WAITING
    assert_raises(
        ValueError,
        lambda: request.transition(RequestState.DECODING, now=10.1),
    )
    request.transition(RequestState.PREFILLING, now=11.0)
    request.transition(RequestState.DECODING, now=12.0)
    request.append_token(9, now=13.0)
    assert request.all_tokens == (1, 2, 9)
    assert request.first_token_at == 13.0
    assert request.remaining_tokens == 0
    assert_raises(RuntimeError, lambda: request.append_token(10, now=14.0))
    request.transition(RequestState.FINISHED, now=15.0)
    assert request.is_terminal
    assert request.finished_at == 15.0


def test_token_budget_scheduler_distinguishes_prefill_and_decode() -> None:
    long_prefill = Request("long", (1, 2, 3, 4), 1, created_at=0.0)
    decode = decoding_request("decode", created_at=1.0)
    short_prefill = Request("short", (5, 6), 1, created_at=2.0)
    scheduled = TokenBudgetScheduler(5).schedule([short_prefill, decode, long_prefill])
    assert [(item.request.request_id, item.phase, item.token_cost) for item in scheduled] == [
        ("long", ExecutionPhase.PREFILL, 4),
        ("decode", ExecutionPhase.DECODE, 1),
    ]


def test_continuous_batching_refills_completed_slot() -> None:
    batcher = ContinuousBatcher(max_batch_size=2)
    first = decoding_request("first")
    second = decoding_request("second")
    third = decoding_request("third")
    for request in (first, second, third):
        batcher.submit(request)
    assert [request.request_id for request in batcher.iteration()] == ["first", "second"]
    first.transition(RequestState.FINISHED, now=1.0)
    assert [request.request_id for request in batcher.iteration()] == ["second", "third"]


def test_chunked_prefill_respects_chunk_and_iteration_budget() -> None:
    planner = ChunkedPrefillPlanner(chunk_size=3)
    chunks = planner.chunks((0, 1, 2, 3, 4, 5, 6))
    assert [chunk.token_ids for chunk in chunks] == [(0, 1, 2), (3, 4, 5), (6,)]
    assert [chunk.is_last for chunk in chunks] == [False, False, True]
    budgeted = planner.next_chunk((0, 1, 2, 3), cursor=1, token_budget=2)
    assert budgeted is not None
    assert budgeted.token_ids == (1, 2)
    assert planner.next_chunk((0, 1), cursor=2) is None


def test_block_allocator_and_paged_kv_cache_reclaim_blocks() -> None:
    allocator = BlockAllocator(total_blocks=3)
    cache = PagedKVCache(allocator, block_size=2)
    cache.append("sequence", (1.0, 0.0), (10.0, 0.0))
    cache.append("sequence", (0.0, 1.0), (0.0, 10.0))
    cache.append("sequence", (1.0, 1.0), (5.0, 5.0))
    assert cache.block_table("sequence") == (0, 1)
    assert [entry.value for entry in cache.entries("sequence")] == [
        (10.0, 0.0),
        (0.0, 10.0),
        (5.0, 5.0),
    ]
    assert allocator.free_count == 1
    cache.free("sequence")
    assert allocator.free_count == 3
    assert allocator.blocks_for("sequence") == ()


def test_prefix_cache_longest_match_refcounts_and_lru_eviction() -> None:
    cache: RadixPrefixCache[str] = RadixPrefixCache(capacity_entries=2)
    cache.insert((1, 2), "short")
    cache.insert((1, 2, 3), "long")
    match = cache.longest_prefix((1, 2, 3, 4), acquire=True)
    assert match is not None
    assert (match.token_ids, match.value, match.refcount) == ((1, 2, 3), "long", 1)
    cache.insert((9,), "new")
    assert cache.longest_prefix((1, 2, 8)) is None
    assert cache.longest_prefix((1, 2, 3)) is not None
    cache.release((1, 2, 3))
    assert cache.evict_one() == (9,)


def test_cache_aware_scheduler_prioritizes_and_discounts_prefix_hits() -> None:
    cache: RadixPrefixCache[str] = RadixPrefixCache(capacity_entries=2)
    cache.insert((1, 2, 3), "kv")
    cold = Request("cold", (8, 9, 10), 1, created_at=0.0)
    warm = Request("warm", (1, 2, 3, 4), 1, created_at=1.0)
    decisions = CacheAwareScheduler(cache, max_batch_tokens=2).schedule([cold, warm])
    actual = [
        (item.request.request_id, item.cached_tokens, item.uncached_token_cost)
        for item in decisions
    ]
    assert actual == [("warm", 3, 1)]


def test_metrics_compute_ttft_inter_token_latency_and_throughput() -> None:
    metrics = MetricsCollector()
    metrics.record_submitted("r", 0.0)
    metrics.record_started("r", 1.0)
    metrics.record_token("r", 2.0)
    metrics.record_token("r", 3.0)
    metrics.record_finished("r", 4.0)
    request_metrics = metrics.for_request("r")
    assert request_metrics.ttft == 2.0
    assert request_metrics.inter_token_latencies == (1.0,)
    snapshot = metrics.snapshot()
    assert snapshot.completed_requests == 1
    assert snapshot.generated_tokens == 2
    assert snapshot.mean_ttft == 2.0
    assert snapshot.mean_inter_token_latency == 1.0
    assert abs(snapshot.throughput_tokens_per_second - 2 / 3) < 1e-12


def test_synchronous_engine_runs_callback_streams_and_refills_batch() -> None:
    clock_value = 0.0

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    calls: list[tuple[tuple[int, ...], ExecutionPhase]] = []

    def model(tokens: tuple[int, ...], phase: ExecutionPhase) -> int:
        calls.append((tokens, phase))
        return 10 if phase is ExecutionPhase.PREFILL else tokens[-1] + 1

    engine = SynchronousEngine(
        model,
        max_batch_size=1,
        max_batch_tokens=4,
        clock=clock,
    )
    first = engine.submit((1, 2), 2, request_id="first")
    second = engine.submit((3,), 1, request_id="second")
    assert engine.run_until_complete() == {"first": (10, 11), "second": (10,)}
    assert first.state is RequestState.FINISHED
    assert second.state is RequestState.FINISHED
    assert [phase for _, phase in calls] == [
        ExecutionPhase.PREFILL,
        ExecutionPhase.DECODE,
        ExecutionPhase.PREFILL,
    ]
    events = engine.events.drain("first")
    assert [event.event_type for event in events] == [
        EventType.START,
        EventType.TOKEN,
        EventType.TOKEN,
        EventType.FINISH,
    ]
    assert [event.token_id for event in events if event.event_type is EventType.TOKEN] == [
        10,
        11,
    ]
