"""Paged KV cache built on fixed-size logical blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mini_infer.engine.block_allocator import BlockAllocator


@dataclass(frozen=True, slots=True)
class KVEntry:
    key: tuple[float, ...]
    value: tuple[float, ...]


class PagedKVCache:
    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.allocator = allocator
        self.block_size = block_size
        self._tables: dict[str, list[int]] = {}
        self._blocks: dict[int, list[KVEntry]] = {}

    def append(self, sequence_id: str, key: Sequence[float], value: Sequence[float]) -> None:
        if not key or len(key) != len(value):
            raise ValueError("key and value must be non-empty and have equal dimensions")
        table = self._tables.setdefault(sequence_id, [])
        if not table or len(self._blocks[table[-1]]) == self.block_size:
            block_id = self.allocator.allocate(sequence_id)[0]
            table.append(block_id)
            self._blocks[block_id] = []
        self._blocks[table[-1]].append(KVEntry(tuple(key), tuple(value)))

    def entries(self, sequence_id: str) -> tuple[KVEntry, ...]:
        table = self._tables.get(sequence_id, [])
        return tuple(entry for block_id in table for entry in self._blocks[block_id])

    def block_table(self, sequence_id: str) -> tuple[int, ...]:
        return tuple(self._tables.get(sequence_id, []))

    def free(self, sequence_id: str) -> None:
        table = self._tables.pop(sequence_id, None)
        if table is None:
            raise KeyError(f"unknown sequence_id: {sequence_id!r}")
        for block_id in table:
            del self._blocks[block_id]
        self.allocator.release(sequence_id, tuple(table))
