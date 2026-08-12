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


@dataclass(slots=True)
class WorkerBatch:
    batch_id: int
    items: list[WorkerItem]


@dataclass(slots=True)
class WorkerResult:
    batch_id: int
    rank: int
    sampled_token_ids: dict[str, int]
    error: str | None = None


@dataclass(slots=True)
class WorkerCommand:
    batch: WorkerBatch | None = None
    shutdown: bool = False
