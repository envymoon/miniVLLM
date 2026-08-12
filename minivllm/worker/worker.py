from __future__ import annotations

from multiprocessing.queues import Queue

from ..config import EngineConfig
from ..protocol import WorkerCommand, WorkerResult
from .model_runner import BootstrapModelRunner
from .operator_runtime import create_kernel_runtime


def worker_process_main(
    rank: int,
    config: EngineConfig,
    command_queue: Queue,
    result_queue: Queue,
) -> None:
    try:
        runtime = create_kernel_runtime(
            config.worker_backend,
            config.num_gpu_blocks,
            config.block_size,
            device_index=config.data_parallel_rank,
        )
        runner = BootstrapModelRunner(runtime, config.block_size)
    except Exception as error:
        result_queue.put(WorkerResult(-1, rank, {}, error=f"worker init: {error}"))
        return

    while True:
        command: WorkerCommand = command_queue.get()
        if command.shutdown:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
            return
        if command.batch is None:
            continue
        sampled: dict[str, int] = {}
        try:
            for item in command.batch.items:
                token_id = runner.execute(item)
                if token_id is not None:
                    sampled[item.request_id] = token_id
            result_queue.put(
                WorkerResult(command.batch.batch_id, rank, sampled_token_ids=sampled)
            )
        except Exception as error:
            result_queue.put(
                WorkerResult(
                    command.batch.batch_id,
                    rank,
                    sampled_token_ids={},
                    error=f"worker {rank}: {error}",
                )
            )
