from __future__ import annotations

from dataclasses import dataclass, field


class BlockPool:
    """Physical KV block allocator. Each Worker/rank owns one instance."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks
        self._free = list(reversed(range(num_blocks)))
        self._used: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise MemoryError("paged KV cache has no free physical blocks")
        blocks = [self._free.pop() for _ in range(count)]
        self._used.update(blocks)
        return blocks

    def free(self, blocks: list[int]) -> None:
        for block in blocks:
            if block not in self._used:
                raise RuntimeError(f"physical block {block} is not allocated")
            self._used.remove(block)
            self._free.append(block)


@dataclass(slots=True)
class BlockTable:
    physical_blocks: list[int] = field(default_factory=list)


class KVCacheManager:
    """Engine-side block tables; Worker-side storage is managed by KernelRuntime."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_blocks)
        self.tables: dict[str, BlockTable] = {}

    def create(self, request_id: str) -> None:
        if request_id in self.tables:
            raise RuntimeError(f"request {request_id} already has a block table")
        self.tables[request_id] = BlockTable()

    def blocks_needed(self, request_id: str, target_tokens: int) -> int:
        table = self.tables[request_id]
        required = (target_tokens + self.block_size - 1) // self.block_size
        return max(0, required - len(table.physical_blocks))

    def can_reserve(self, request_id: str, target_tokens: int) -> bool:
        return self.blocks_needed(request_id, target_tokens) <= self.pool.num_free_blocks

    def reserve(self, request_id: str, target_tokens: int) -> None:
        count = self.blocks_needed(request_id, target_tokens)
        self.tables[request_id].physical_blocks.extend(self.pool.allocate(count))

    def slots(self, request_id: str, start: int, count: int) -> list[int]:
        table = self.tables[request_id].physical_blocks
        slots: list[int] = []
        for position in range(start, start + count):
            logical_block, offset = divmod(position, self.block_size)
            slots.append(table[logical_block] * self.block_size + offset)
        return slots

    def block_table(self, request_id: str) -> list[int]:
        return list(self.tables[request_id].physical_blocks)

    def release(self, request_id: str) -> None:
        table = self.tables.pop(request_id, None)
        if table is not None:
            self.pool.free(table.physical_blocks)
