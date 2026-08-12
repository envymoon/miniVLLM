from __future__ import annotations

from typing import Protocol

from ..config import EngineConfig
from ..tokenizer import CharacterTokenizer
from .batch import GPUInputBatch
from .operator_runtime import KernelRuntime, create_kernel_runtime


class ModelRunner(Protocol):
    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]: ...

    def close(self) -> None: ...


class BootstrapModelRunner:
    """Deterministic bootstrap runner used until model layers are connected."""

    _response = " mini-vLLM"

    def __init__(self, runtime: KernelRuntime, block_size: int) -> None:
        self.runtime = runtime
        self.block_size = block_size
        self.tokenizer = CharacterTokenizer()
        self._response_tokens = self.tokenizer.encode(self._response)

    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]:
        """Run one flattened batch rather than invoking the runner per request."""
        self.runtime.append_kv(batch.input_ids, batch.slot_mapping)
        self.runtime.attention_batch(batch)
        return self._sample(batch)

    def _sample(self, batch: GPUInputBatch) -> dict[str, int]:
        sampled: dict[str, int] = {}
        for index, request_id in enumerate(batch.request_ids):
            if batch.sample_indices[index] < 0:
                continue
            generated_count = batch.generated_counts[index]
            if generated_count >= len(self._response_tokens):
                sampled[request_id] = self.tokenizer.eos_token_id
            else:
                sampled[request_id] = self._response_tokens[generated_count]
        return sampled

    def close(self) -> None:
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()


class CudaBootstrapModelRunner(BootstrapModelRunner):
    supports_cudagraph = True

    def execute_cudagraph(
        self, batch: GPUInputBatch, capture_size: int
    ) -> dict[str, int]:
        execute_graph = getattr(self.runtime, "execute_decode_graph")
        execute_graph(batch, capture_size)
        return self._sample(batch)


def create_model_runner(config: EngineConfig) -> ModelRunner:
    if config.model_path is not None:
        from ..model_executor.llama import LlamaModelRunner

        return LlamaModelRunner(
            model_path=config.model_path,
            num_gpu_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
            max_model_len=config.max_model_len,
            device=config.device,
            dtype=config.dtype,
            data_parallel_rank=config.data_parallel_rank,
            max_num_seqs=config.max_num_seqs,
            attention_backend=config.attention_backend,
        )
    runtime = create_kernel_runtime(
        config.worker_backend,
        config.num_gpu_blocks,
        config.block_size,
        device_index=config.data_parallel_rank,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
    )
    runner_type = (
        CudaBootstrapModelRunner
        if config.worker_backend == "cuda"
        else BootstrapModelRunner
    )
    return runner_type(runtime, config.block_size)
