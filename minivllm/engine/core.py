from __future__ import annotations

import multiprocessing as mp
import queue
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue

from ..config import EngineConfig
from ..protocol import (
    CommandType,
    EngineCommand,
    FinishReason,
    RequestOutput,
    WorkerBatch,
    WorkerCommand,
    WorkerResult,
)
from ..scheduler import Scheduler
from ..tokenizer import CharacterTokenizer
from ..worker.worker import worker_process_main


class EngineCore:
    """The v1-style scheduling loop. This object lives in its own process."""

    def __init__(
        self,
        config: EngineConfig,
        context: BaseContext,
        command_queue: Queue,
        output_queue: Queue,
    ) -> None:
        self.config = config
        self.context = context
        self.command_queue = command_queue
        self.output_queue = output_queue
        self.scheduler = Scheduler(config)
        self.tokenizer = CharacterTokenizer()
        self.worker_result_queue = context.Queue()
        self.worker_queues = [context.Queue() for _ in range(config.data_parallel_size)]
        self.workers = [
            context.Process(
                target=worker_process_main,
                args=(rank, config, self.worker_queues[rank], self.worker_result_queue),
                name=f"minivllm-worker-dp{config.data_parallel_rank}",
            )
            for rank in range(config.data_parallel_size)
        ]
        self._batch_id = 0
        self._shutdown = False

    def _emit_error(self, request_id: str, error: Exception | str) -> None:
        self.output_queue.put(
            RequestOutput(
                request_id=request_id,
                token_ids=[],
                text="",
                finished=True,
                finish_reason=FinishReason.ERROR,
                error=str(error),
            )
        )

    def _handle_command(self, command: EngineCommand) -> None:
        if command.type is CommandType.SHUTDOWN:
            self._shutdown = True
        elif command.type is CommandType.ADD and command.request is not None:
            try:
                self.scheduler.add_request(command.request)
            except Exception as error:
                self._emit_error(command.request.request_id, error)
        elif command.type is CommandType.ABORT and command.request_id is not None:
            if self.scheduler.abort(command.request_id):
                self.output_queue.put(
                    RequestOutput(
                        request_id=command.request_id,
                        token_ids=[],
                        text="",
                        finished=True,
                        finish_reason=FinishReason.ABORT,
                    )
                )

    def _drain_commands(self, block: bool = False) -> None:
        if block:
            try:
                self._handle_command(
                    self.command_queue.get(timeout=self.config.scheduler_idle_timeout_s)
                )
            except queue.Empty:
                return
        while True:
            try:
                self._handle_command(self.command_queue.get_nowait())
            except queue.Empty:
                return

    def _execute_schedule(self) -> None:
        schedule = self.scheduler.schedule()
        if not schedule.by_rank:
            return
        self._batch_id += 1
        for rank, items in schedule.by_rank.items():
            self.worker_queues[rank].put(
                WorkerCommand(batch=WorkerBatch(self._batch_id, items))
            )

        results: dict[int, WorkerResult] = {}
        while len(results) < len(schedule.by_rank):
            result: WorkerResult = self.worker_result_queue.get()
            if result.batch_id == -1:
                raise RuntimeError(result.error)
            if result.batch_id == self._batch_id:
                results[result.rank] = result

        for result in results.values():
            if result.error:
                for item in schedule.by_rank[result.rank]:
                    self._emit_error(item.request_id, result.error)
                    self.scheduler.finish(item.request_id)
                continue
            for item in schedule.by_rank[result.rank]:
                if item.request_id not in self.scheduler.running:
                    continue
                self.scheduler.update_computed(
                    item.request_id, schedule.scheduled_counts[item.request_id]
                )
                token_id = result.sampled_token_ids.get(item.request_id)
                if token_id is None:
                    continue
                finish_reason = self.scheduler.append_token(item.request_id, token_id)
                text = "" if token_id == self.tokenizer.eos_token_id else self.tokenizer.decode([token_id])
                self.output_queue.put(
                    RequestOutput(
                        request_id=item.request_id,
                        token_ids=[] if token_id == self.tokenizer.eos_token_id else [token_id],
                        text=text,
                        finished=finish_reason is not None,
                        finish_reason=finish_reason,
                    )
                )
                if finish_reason is not None:
                    self.scheduler.finish(item.request_id)

    def run(self) -> None:
        for worker in self.workers:
            worker.start()
        try:
            while not self._shutdown:
                self._drain_commands(block=not self.scheduler.has_requests)
                if self._shutdown:
                    break
                self._execute_schedule()
        except Exception as error:
            for request_id in list(self.scheduler.running):
                self._emit_error(request_id, error)
            for request in list(self.scheduler.waiting):
                self._emit_error(request.request_id, error)
        finally:
            for worker_queue in self.worker_queues:
                worker_queue.put(WorkerCommand(shutdown=True))
            for worker in self.workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()


def engine_core_process_main(
    config: EngineConfig,
    command_queue: Queue,
    output_queue: Queue,
) -> None:
    context = mp.get_context(config.process_start_method)
    EngineCore(config, context, command_queue, output_queue).run()
