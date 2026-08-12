from __future__ import annotations

from dataclasses import dataclass

from ..protocol import WorkerBatch


@dataclass(frozen=True, slots=True)
class BatchDescriptor:
    """Minimal execution-shape key shared by eager and CUDA Graph paths."""

    num_tokens: int
    num_requests: int
    uniform_decode: bool


@dataclass(slots=True)
class GPUInputBatch:
    """Flattened scheduler output consumed by one ModelRunner invocation."""

    batch_id: int
    request_ids: tuple[str, ...]
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    query_start_loc: list[int]
    seq_lens: list[int]
    block_tables: list[list[int]]
    context_slot_mapping: list[int]
    context_start_loc: list[int]
    is_decode: list[bool]
    sample_indices: list[int]
    generated_counts: list[int]
    temperatures: list[float]

    @classmethod
    def from_worker_batch(
        cls, worker_batch: WorkerBatch, block_size: int
    ) -> GPUInputBatch:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not worker_batch.items:
            raise ValueError("worker batch must contain at least one request")

        request_ids: list[str] = []
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        query_start_loc = [0]
        seq_lens: list[int] = []
        raw_block_tables: list[list[int]] = []
        context_slot_mapping: list[int] = []
        context_start_loc = [0]
        is_decode: list[bool] = []
        sample_indices: list[int] = []
        generated_counts: list[int] = []
        temperatures: list[float] = []

        for item in worker_batch.items:
            if not item.token_ids:
                raise ValueError(f"request {item.request_id} scheduled no tokens")
            if len(item.token_ids) != len(item.slot_mapping):
                raise ValueError("token_ids and slot_mapping must have equal length")
            if item.start_pos < 0 or item.seq_len != item.start_pos + len(item.token_ids):
                raise ValueError(f"invalid token range for request {item.request_id}")
            required_blocks = (item.seq_len + block_size - 1) // block_size
            if len(item.block_table) < required_blocks:
                raise ValueError(f"incomplete block table for request {item.request_id}")

            request_ids.append(item.request_id)
            input_ids.extend(item.token_ids)
            positions.extend(range(item.start_pos, item.seq_len))
            slot_mapping.extend(item.slot_mapping)
            query_start_loc.append(len(input_ids))
            seq_lens.append(item.seq_len)
            raw_block_tables.append(item.block_table[:required_blocks])

            context_slot_mapping.extend(
                item.block_table[position // block_size] * block_size
                + position % block_size
                for position in range(item.seq_len)
            )
            context_start_loc.append(len(context_slot_mapping))
            decode = len(item.token_ids) == 1 and item.generated_count > 0
            is_decode.append(decode)
            sample_indices.append(len(input_ids) - 1 if item.should_sample else -1)
            generated_counts.append(item.generated_count)
            temperatures.append(item.temperature)

        max_blocks = max(len(table) for table in raw_block_tables)
        block_tables = [
            table + [-1] * (max_blocks - len(table)) for table in raw_block_tables
        ]
        return cls(
            batch_id=worker_batch.batch_id,
            request_ids=tuple(request_ids),
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_tables,
            context_slot_mapping=context_slot_mapping,
            context_start_loc=context_start_loc,
            is_decode=is_decode,
            sample_indices=sample_indices,
            generated_counts=generated_counts,
            temperatures=temperatures,
        )

    @property
    def num_requests(self) -> int:
        return len(self.request_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def num_decode_tokens(self) -> int:
        return sum(
            self.query_start_loc[index + 1] - self.query_start_loc[index]
            for index, decode in enumerate(self.is_decode)
            if decode
        )

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_tokens

    @property
    def descriptor(self) -> BatchDescriptor:
        return BatchDescriptor(
            num_tokens=self.num_tokens,
            num_requests=self.num_requests,
            uniform_decode=all(self.is_decode),
        )
