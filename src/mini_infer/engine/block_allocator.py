"""Fixed-size block allocation."""

from __future__ import annotations

from collections import defaultdict


class BlockAllocator:
    def __init__(self, total_blocks: int) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        self.total_blocks = total_blocks
        self._free = list(range(total_blocks - 1, -1, -1))
        self._owners: dict[str, set[int]] = defaultdict(set)

    @property
    def free_count(self) -> int:
        return len(self._free)

    def allocate(self, owner: str, count: int = 1) -> tuple[int, ...]:
        if not owner:
            raise ValueError("owner must not be empty")
        if count <= 0:
            raise ValueError("count must be positive")
        if count > self.free_count:
            raise MemoryError(f"requested {count} blocks, only {self.free_count} are free")
        blocks = tuple(self._free.pop() for _ in range(count))
        self._owners[owner].update(blocks)
        return blocks

    def release(self, owner: str, blocks: tuple[int, ...] | None = None) -> None:
        owned = self._owners.get(owner)
        if not owned:
            raise KeyError(f"owner has no allocated blocks: {owner!r}")
        selected = set(owned if blocks is None else blocks)
        if not selected.issubset(owned):
            raise ValueError("cannot release blocks not owned by owner")
        owned.difference_update(selected)
        if not owned:
            del self._owners[owner]
        self._free.extend(sorted(selected, reverse=True))

    def blocks_for(self, owner: str) -> tuple[int, ...]:
        return tuple(sorted(self._owners.get(owner, set())))
