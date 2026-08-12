from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .config import SamplingParams


class CommandType(Enum):
    ADD = auto()
    ABORT = auto()
    SHUTDOWN = auto()


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"


@dataclass(slots=True)
class EngineRequest:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams


@dataclass(slots=True)
class EngineCommand:
    type: CommandType
    request: EngineRequest | None = None
    request_id: str | None = None


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    token_ids: list[int]
    text: str
    finished: bool
    finish_reason: FinishReason | None = None
    error: str | None = None
    metrics: BatchMetrics | None = None
    request_metrics: RequestMetrics | None = None


@dataclass(slots=True)
class WorkerItem:
    request_id: str
    token_ids: list[int]
    start_pos: int
    slot_mapping: list[int]
    block_table: list[int]
    seq_len: int
    generated_count: int
    should_sample: bool
    temperature: float = 0.0


@dataclass(slots=True)
class WorkerBatch:
    batch_id: int
    items: list[WorkerItem]


@dataclass(frozen=True, slots=True)
class BatchMetrics:
    batch_id: int
    rank: int
    num_requests: int
    num_scheduled_tokens: int
    num_prefill_tokens: int
    num_decode_tokens: int
    execute_time_s: float
    execution_mode: str = "eager"


@dataclass(frozen=True, slots=True)
class EngineMetricsSnapshot:
    total_batches: int
    total_scheduled_tokens: int
    total_prefill_tokens: int
    total_decode_tokens: int
    total_worker_execute_time_s: float

    @property
    def average_tokens_per_batch(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_scheduled_tokens / self.total_batches


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    time_to_first_token_s: float | None
    inter_token_latency_s: float | None
    end_to_end_latency_s: float | None
    num_output_tokens: int


@dataclass(slots=True)
class WorkerResult:
    batch_id: int
    rank: int
    sampled_token_ids: dict[str, int]
    error: str | None = None
    metrics: BatchMetrics | None = None


@dataclass(slots=True)
class WorkerCommand:
    batch: WorkerBatch | None = None
    shutdown: bool = False
