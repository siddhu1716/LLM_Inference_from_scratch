"""CUDA graph concept simulation with a static-shape callable cache.

No CUDA work occurs here, and this class does not claim a speedup. It only models
the capture-once/replay-many control flow used by real CUDA graph integrations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

ResultT = TypeVar("ResultT")


class StaticShapeGraphCache(Generic[ResultT]):
    def __init__(self) -> None:
        self._graphs: dict[tuple[int, ...], Callable[..., ResultT]] = {}
        self.capture_count = 0
        self.replay_count = 0

    def execute(
        self,
        shape: tuple[int, ...],
        capture: Callable[[], Callable[..., ResultT]],
        *args: object,
        **kwargs: object,
    ) -> ResultT:
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("shape dimensions must be positive")
        graph = self._graphs.get(shape)
        if graph is None:
            graph = capture()
            if not callable(graph):
                raise TypeError("capture must return a callable")
            self._graphs[shape] = graph
            self.capture_count += 1
        else:
            self.replay_count += 1
        return graph(*args, **kwargs)

    @property
    def cached_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(self._graphs))
