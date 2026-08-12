from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import replace
from multiprocessing.queues import Queue

from ..config import EngineConfig
from ..protocol import BatchMetrics, CommandType, EngineCommand, RequestOutput
from .core import engine_core_process_main


class DPCoordinator:
    """Load-balances requests across one EngineCore process per DP rank."""

    def __init__(
        self, config: EngineConfig, command_queue: Queue, output_queue: Queue
    ) -> None:
        self.config = config
        self.command_queue = command_queue
        self.output_queue = output_queue
        self.context = mp.get_context(config.process_start_method)
        self.rank_command_queues = [
            self.context.Queue() for _ in range(config.data_parallel_size)
        ]
        self.rank_output_queues = [
            self.context.Queue() for _ in range(config.data_parallel_size)
        ]
        self.engine_cores = []
        for rank in range(config.data_parallel_size):
            rank_config = replace(
                config, data_parallel_size=1, data_parallel_rank=rank
            )
            self.engine_cores.append(
                self.context.Process(
                    target=engine_core_process_main,
                    args=(
                        rank_config,
                        self.rank_command_queues[rank],
                        self.rank_output_queues[rank],
                    ),
                    name=f"minivllm-engine-core-dp{rank}",
                )
            )
        self.request_ranks: dict[str, int] = {}
        self.rank_load = [0] * config.data_parallel_size
        self._shutdown = False

    def _handle_command(self, command: EngineCommand) -> None:
        if command.type is CommandType.SHUTDOWN:
            self._shutdown = True
            return
        if command.type is CommandType.ADD and command.request is not None:
            rank = min(range(self.config.data_parallel_size), key=self.rank_load.__getitem__)
            self.request_ranks[command.request.request_id] = rank
            self.rank_load[rank] += 1
            self.rank_command_queues[rank].put(command)
            return
        if command.type is CommandType.ABORT and command.request_id is not None:
            rank = self.request_ranks.get(command.request_id)
            if rank is not None:
                self.rank_command_queues[rank].put(command)

    def _drain_parent_commands(self) -> None:
        try:
            self._handle_command(
                self.command_queue.get(timeout=self.config.scheduler_idle_timeout_s)
            )
        except queue.Empty:
            pass
        while True:
            try:
                self._handle_command(self.command_queue.get_nowait())
            except queue.Empty:
                return

    def _drain_rank_outputs(self) -> None:
        for rank, rank_queue in enumerate(self.rank_output_queues):
            while True:
                try:
                    output: RequestOutput | BatchMetrics = rank_queue.get_nowait()
                except queue.Empty:
                    break
                self.output_queue.put(output)
                if isinstance(output, BatchMetrics):
                    continue
                if output.finished:
                    self.request_ranks.pop(output.request_id, None)
                    self.rank_load[rank] = max(0, self.rank_load[rank] - 1)

    def run(self) -> None:
        for engine_core in self.engine_cores:
            engine_core.start()
        try:
            while not self._shutdown:
                self._drain_parent_commands()
                self._drain_rank_outputs()
        finally:
            shutdown = EngineCommand(CommandType.SHUTDOWN)
            for rank_queue in self.rank_command_queues:
                rank_queue.put(shutdown)
            for engine_core in self.engine_cores:
                engine_core.join(timeout=10)
                if engine_core.is_alive():
                    engine_core.terminate()
                    engine_core.join(timeout=2)


def dp_coordinator_process_main(
    config: EngineConfig, command_queue: Queue, output_queue: Queue
) -> None:
    DPCoordinator(config, command_queue, output_queue).run()
