from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .cache import KVCacheManager
from .config import EngineConfig, SamplingParams
from .protocol import EngineRequest, FinishReason, WorkerItem
from .tokenizer import CharacterTokenizer


@dataclass(slots=True)
class RequestState:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    rank: int | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids


@dataclass(slots=True)
class Schedule:
    by_rank: dict[int, list[WorkerItem]]
    scheduled_counts: dict[str, int]


class Scheduler:
    """A small vLLM v1-style token scheduler with chunked prefill."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.waiting: deque[RequestState] = deque()
        self.running: OrderedDict[str, RequestState] = OrderedDict()
        self.cache_manager = KVCacheManager(
            config.num_gpu_blocks, config.block_size
        )
        self.tokenizer = CharacterTokenizer()

    def add_request(self, request: EngineRequest) -> None:
        if request.request_id in self.running or any(
            item.request_id == request.request_id for item in self.waiting
        ):
            raise ValueError(f"duplicate request id: {request.request_id}")
        context_limit = min(
            self.config.max_model_len,
            self.config.num_gpu_blocks * self.config.block_size,
        )
        if len(request.prompt_token_ids) >= context_limit:
            raise ValueError(
                "prompt leaves no generation capacity in this rank's KV cache"
            )
        self.waiting.append(
            RequestState(
                request_id=request.request_id,
                prompt=request.prompt,
                prompt_token_ids=request.prompt_token_ids,
                sampling_params=request.sampling_params,
            )
        )

    def _admit_waiting(self) -> None:
        attempts = len(self.waiting)
        while self.waiting and len(self.running) < self.config.max_num_seqs and attempts:
            request = self.waiting.popleft()
            attempts -= 1
            manager = self.cache_manager
            manager.create(request.request_id)
            first_target = min(len(request.prompt_token_ids), self.config.max_num_batched_tokens)
            if not manager.can_reserve(request.request_id, first_target):
                manager.release(request.request_id)
                self.waiting.append(request)
                continue
            request.rank = 0
            self.running[request.request_id] = request

    def schedule(self) -> Schedule:
        self._admit_waiting()
        budget = self.config.max_num_batched_tokens
        by_rank: dict[int, list[WorkerItem]] = {}
        scheduled_counts: dict[str, int] = {}

        for request in list(self.running.values()):
            if budget == 0:
                break
            all_tokens = request.all_token_ids
            remaining = len(all_tokens) - request.num_computed_tokens
            if remaining <= 0:
                continue
            count = min(remaining, budget)
            target = request.num_computed_tokens + count
            manager = self.cache_manager
            if not manager.can_reserve(request.request_id, target):
                continue
            manager.reserve(request.request_id, target)
            start = request.num_computed_tokens
            item = WorkerItem(
                request_id=request.request_id,
                token_ids=all_tokens[start:target],
                start_pos=start,
                slot_mapping=manager.slots(request.request_id, start, count),
                block_table=manager.block_table(request.request_id),
                seq_len=target,
                generated_count=len(request.output_token_ids),
                should_sample=target == len(all_tokens),
            )
            by_rank.setdefault(request.rank, []).append(item)  # type: ignore[arg-type]
            scheduled_counts[request.request_id] = count
            budget -= count
        return Schedule(by_rank=by_rank, scheduled_counts=scheduled_counts)

    def update_computed(self, request_id: str, count: int) -> None:
        self.running[request_id].num_computed_tokens += count

    def append_token(self, request_id: str, token_id: int) -> FinishReason | None:
        request = self.running[request_id]
        if token_id == 1:
            return FinishReason.STOP
        request.output_token_ids.append(token_id)
        output_text = self.tokenizer.decode(request.output_token_ids)
        if any(output_text.endswith(stop) for stop in request.sampling_params.stop):
            return FinishReason.STOP
        if len(request.output_token_ids) >= request.sampling_params.max_tokens:
            return FinishReason.LENGTH
        context_limit = min(
            self.config.max_model_len,
            self.config.num_gpu_blocks * self.config.block_size,
        )
        if request.num_computed_tokens + 1 >= context_limit:
            return FinishReason.LENGTH
        return None

    def finish(self, request_id: str) -> RequestState | None:
        request = self.running.pop(request_id, None)
        if request is not None:
            self.cache_manager.release(request_id)
        return request

    def abort(self, request_id: str) -> bool:
        if request_id in self.running:
            self.finish(request_id)
            return True
        for request in list(self.waiting):
            if request.request_id == request_id:
                self.waiting.remove(request)
                return True
        return False

    @property
    def has_requests(self) -> bool:
        return bool(self.waiting or self.running)
