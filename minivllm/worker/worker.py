from __future__ import annotations

import time
from multiprocessing.queues import Queue

from ..config import EngineConfig
from ..protocol import BatchMetrics, WorkerCommand, WorkerResult
from .batch import GPUInputBatch
from .cudagraph import CUDAGraphManager
from .model_runner import create_model_runner


def worker_process_main(
    rank: int,
    config: EngineConfig,
    command_queue: Queue,
    result_queue: Queue,
) -> None:
    try:
        runner = create_model_runner(config)
        graph_manager = CUDAGraphManager(
            config.cudagraph_mode,
            tuple(
                size
                for size in config.cudagraph_capture_sizes
                if size <= config.max_num_seqs
            ),
        )
    except Exception as error:
        result_queue.put(WorkerResult(-1, rank, {}, error=f"worker init: {error}"))
        return

    while True:
        command: WorkerCommand = command_queue.get()
        if command.shutdown:
            runner.close()
            return
        if command.batch is None:
            continue
        sampled: dict[str, int] = {}
        try:
            input_batch = GPUInputBatch.from_worker_batch(
                command.batch, config.block_size
            )
            started = time.perf_counter()
            sampled, execution_mode = graph_manager.execute(runner, input_batch)
            execute_time_s = time.perf_counter() - started
            result_queue.put(
                WorkerResult(
                    command.batch.batch_id,
                    rank,
                    sampled_token_ids=sampled,
                    metrics=BatchMetrics(
                        batch_id=command.batch.batch_id,
                        rank=config.data_parallel_rank,
                        num_requests=input_batch.num_requests,
                        num_scheduled_tokens=input_batch.num_tokens,
                        num_prefill_tokens=input_batch.num_prefill_tokens,
                        num_decode_tokens=input_batch.num_decode_tokens,
                        execute_time_s=execute_time_s,
                        execution_mode=execution_mode,
                    ),
                )
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
