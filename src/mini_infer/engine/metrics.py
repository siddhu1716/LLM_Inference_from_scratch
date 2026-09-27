"""TTFT, inter-token latency, and throughput metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean


@dataclass(slots=True)
class RequestMetrics:
    submitted_at: float
    started_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None

    @property
    def ttft(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[0] - self.submitted_at

    @property
    def inter_token_latencies(self) -> tuple[float, ...]:
        return tuple(
            current - previous
            for previous, current in zip(
                self.token_times,
                self.token_times[1:],
                strict=False,
            )
        )


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    completed_requests: int
    generated_tokens: int
    mean_ttft: float | None
    mean_inter_token_latency: float | None
    throughput_tokens_per_second: float


class MetricsCollector:
    def __init__(self) -> None:
        self._requests: dict[str, RequestMetrics] = {}

    def record_submitted(self, request_id: str, timestamp: float) -> None:
        if request_id in self._requests:
            raise ValueError(f"metrics already exist for request: {request_id!r}")
        self._requests[request_id] = RequestMetrics(timestamp)

    def record_started(self, request_id: str, timestamp: float) -> None:
        metrics = self._require(request_id)
        if metrics.started_at is not None:
            raise RuntimeError("request start was already recorded")
        metrics.started_at = timestamp

    def record_token(self, request_id: str, timestamp: float) -> None:
        metrics = self._require(request_id)
        if metrics.started_at is None:
            raise RuntimeError("cannot record a token before request start")
        if metrics.token_times and timestamp < metrics.token_times[-1]:
            raise ValueError("token timestamps must be monotonic")
        metrics.token_times.append(timestamp)

    def record_finished(self, request_id: str, timestamp: float) -> None:
        metrics = self._require(request_id)
        if metrics.finished_at is not None:
            raise RuntimeError("request finish was already recorded")
        if metrics.started_at is None or timestamp < metrics.started_at:
            raise ValueError("finish must not precede request start")
        metrics.finished_at = timestamp

    def for_request(self, request_id: str) -> RequestMetrics:
        return self._require(request_id)

    def snapshot(self) -> MetricsSnapshot:
        completed = [
            metrics
            for metrics in self._requests.values()
            if metrics.finished_at is not None
        ]
        ttfts = [metrics.ttft for metrics in completed if metrics.ttft is not None]
        itls = [
            latency
            for metrics in completed
            for latency in metrics.inter_token_latencies
        ]
        generated_tokens = sum(len(metrics.token_times) for metrics in completed)
        starts = [
            metrics.started_at for metrics in completed if metrics.started_at is not None
        ]
        finishes = [
            metrics.finished_at for metrics in completed if metrics.finished_at is not None
        ]
        duration = max(finishes) - min(starts) if starts and finishes else 0.0
        throughput = generated_tokens / duration if duration > 0 else 0.0
        return MetricsSnapshot(
            completed_requests=len(completed),
            generated_tokens=generated_tokens,
            mean_ttft=fmean(ttfts) if ttfts else None,
            mean_inter_token_latency=fmean(itls) if itls else None,
            throughput_tokens_per_second=throughput,
        )

    def _require(self, request_id: str) -> RequestMetrics:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request metrics: {request_id!r}") from exc
