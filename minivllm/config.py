from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if any(not item for item in self.stop):
            raise ValueError("stop strings must not be empty")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Core engine, scheduling, memory, and multiprocessing configuration."""

    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 256
    max_model_len: int = 2048
    block_size: int = 16
    num_gpu_blocks: int = 256
    worker_backend: str = "reference"
    model_path: str | None = None
    tokenizer_path: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    attention_backend: str = "auto"
    cudagraph_mode: str = "none"
    cudagraph_capture_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    process_start_method: str = "spawn"
    scheduler_idle_timeout_s: float = 0.01

    def __post_init__(self) -> None:
        positive = {
            "data_parallel_size": self.data_parallel_size,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "block_size": self.block_size,
            "num_gpu_blocks": self.num_gpu_blocks,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.worker_backend not in {"reference", "cuda"}:
            raise ValueError("worker_backend must be 'reference' or 'cuda'")
        if (
            self.device != "auto"
            and self.device != "cpu"
            and not self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("unsupported model dtype")
        if self.attention_backend not in {"auto", "torch", "custom"}:
            raise ValueError("unsupported attention backend")
        if self.cudagraph_mode not in {"none", "full_decode_only"}:
            raise ValueError("cudagraph_mode must be 'none' or 'full_decode_only'")
        if (
            not self.cudagraph_capture_sizes
            or any(size <= 0 for size in self.cudagraph_capture_sizes)
            or tuple(sorted(set(self.cudagraph_capture_sizes)))
            != self.cudagraph_capture_sizes
        ):
            raise ValueError("cudagraph_capture_sizes must be sorted unique positives")
        if self.data_parallel_rank < 0:
            raise ValueError("data_parallel_rank must be non-negative")
        if self.process_start_method not in {"spawn", "fork", "forkserver"}:
            raise ValueError("unsupported multiprocessing start method")
