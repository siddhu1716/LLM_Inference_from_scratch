"""Radix-style token-prefix cache with refcounts and LRU eviction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


@dataclass(slots=True)
class _Entry(Generic[ValueT]):
    value: ValueT
    refcount: int
    last_access: int


@dataclass(frozen=True, slots=True)
class PrefixMatch(Generic[ValueT]):
    token_ids: tuple[int, ...]
    value: ValueT
    refcount: int

    @property
    def length(self) -> int:
        return len(self.token_ids)


class _Node:
    __slots__ = ("children", "prefix")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.prefix: tuple[int, ...] | None = None


class RadixPrefixCache(Generic[ValueT]):
    def __init__(self, capacity_entries: int) -> None:
        if capacity_entries <= 0:
            raise ValueError("capacity_entries must be positive")
        self.capacity_entries = capacity_entries
        self._root = _Node()
        self._entries: dict[tuple[int, ...], _Entry[ValueT]] = {}
        self._clock = 0

    def __len__(self) -> int:
        return len(self._entries)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def insert(self, token_ids: Sequence[int], value: ValueT) -> None:
        prefix = tuple(token_ids)
        if not prefix:
            raise ValueError("token_ids must not be empty")
        existing = self._entries.get(prefix)
        if existing is not None:
            existing.value = value
            existing.last_access = self._tick()
            return
        while len(self._entries) >= self.capacity_entries:
            self.evict_one()
        node = self._root
        for token_id in prefix:
            node = node.children.setdefault(token_id, _Node())
        node.prefix = prefix
        self._entries[prefix] = _Entry(value, 0, self._tick())

    def longest_prefix(
        self,
        token_ids: Sequence[int],
        *,
        acquire: bool = False,
    ) -> PrefixMatch[ValueT] | None:
        node = self._root
        best: tuple[int, ...] | None = None
        for token_id in token_ids:
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            if node.prefix is not None:
                best = node.prefix
        if best is None:
            return None
        entry = self._entries[best]
        entry.last_access = self._tick()
        if acquire:
            entry.refcount += 1
        return PrefixMatch(best, entry.value, entry.refcount)

    def release(self, token_ids: Sequence[int]) -> None:
        prefix = tuple(token_ids)
        entry = self._entries.get(prefix)
        if entry is None:
            raise KeyError(f"prefix is not cached: {prefix!r}")
        if entry.refcount == 0:
            raise RuntimeError("prefix refcount is already zero")
        entry.refcount -= 1
        entry.last_access = self._tick()

    def evict_one(self) -> tuple[int, ...]:
        candidates = [
            (entry.last_access, prefix)
            for prefix, entry in self._entries.items()
            if entry.refcount == 0
        ]
        if not candidates:
            raise RuntimeError("all prefix-cache entries are pinned")
        _, victim = min(candidates)
        del self._entries[victim]
        self._remove_from_trie(victim)
        return victim

    def _remove_from_trie(self, prefix: tuple[int, ...]) -> None:
        path: list[tuple[_Node, int, _Node]] = []
        node = self._root
        for token_id in prefix:
            child = node.children[token_id]
            path.append((node, token_id, child))
            node = child
        node.prefix = None
        for parent, token_id, child in reversed(path):
            if child.prefix is not None or child.children:
                break
            del parent.children[token_id]
