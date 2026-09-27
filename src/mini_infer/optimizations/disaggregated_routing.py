"""Prefill/decode disaggregation routing simulation.

This models routing decisions only; it is not a distributed runtime or speedup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class WorkerKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    request_id: str
    worker_id: str
    worker_kind: WorkerKind
    transferred_from: str | None = None


class PrefillDecodeRouter:
    def __init__(
        self,
        prefill_workers: Sequence[str],
        decode_workers: Sequence[str],
    ) -> None:
        if not prefill_workers or not decode_workers:
            raise ValueError("prefill and decode worker pools must both be non-empty")
        workers = tuple(prefill_workers) + tuple(decode_workers)
        if any(not worker for worker in workers) or len(set(workers)) != len(workers):
            raise ValueError("worker IDs must be non-empty and unique")
        self._pools = {
            WorkerKind.PREFILL: tuple(prefill_workers),
            WorkerKind.DECODE: tuple(decode_workers),
        }
        self._loads = {worker: 0 for worker in workers}
        self._prefill_assignment: dict[str, str] = {}

    def route(self, request_id: str, kind: WorkerKind) -> RoutingDecision:
        if not request_id:
            raise ValueError("request_id must not be empty")
        worker = min(self._pools[kind], key=lambda item: (self._loads[item], item))
        self._loads[worker] += 1
        if kind is WorkerKind.PREFILL:
            self._prefill_assignment[request_id] = worker
            source = None
        else:
            source = self._prefill_assignment.get(request_id)
        return RoutingDecision(request_id, worker, kind, source)

    def complete(self, worker_id: str) -> None:
        if worker_id not in self._loads:
            raise KeyError(f"unknown worker_id: {worker_id!r}")
        if self._loads[worker_id] == 0:
            raise RuntimeError("worker has no routed work to complete")
        self._loads[worker_id] -= 1

    def load(self, worker_id: str) -> int:
        try:
            return self._loads[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown worker_id: {worker_id!r}") from exc
