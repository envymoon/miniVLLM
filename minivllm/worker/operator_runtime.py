from __future__ import annotations

import ctypes
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .batch import GPUInputBatch


class KernelRuntime(ABC):
    """The narrow boundary between the vLLM system and attention kernels."""

    @abstractmethod
    def append_kv(self, token_ids: list[int], slots: list[int]) -> None: ...

    @abstractmethod
    def flash_attention_prefill(self, slots: list[int]) -> None: ...

    @abstractmethod
    def paged_decode_attention(
        self, block_table: list[int], seq_len: int, block_size: int
    ) -> None: ...

    @abstractmethod
    def attention_batch(self, batch: GPUInputBatch) -> None:
        """Execute all prefill/decode attention work represented by one batch."""
        ...


class ReferenceKernelRuntime(KernelRuntime):
    """CPU reference for the same append/prefill/decode calls as the CUDA ABI.

    It computes tiny synthetic vectors because model weights are not connected yet;
    every scheduled token still traverses the real KV slot mapping
    and attention control flow.
    """

    def __init__(self, head_dim: int = 8) -> None:
        self.head_dim = head_dim
        self.k_cache: dict[int, list[float]] = {}
        self.v_cache: dict[int, list[float]] = {}
        self.last_attention_output: list[float] = []
        self.last_batch_attention_outputs: list[list[float]] = []

    def _vector(self, token_id: int, phase: float) -> list[float]:
        return [math.sin(token_id * 0.01 + dim * phase) for dim in range(self.head_dim)]

    def append_kv(self, token_ids: list[int], slots: list[int]) -> None:
        if len(token_ids) != len(slots):
            raise ValueError("token_ids and slot_mapping must have equal length")
        for token_id, slot in zip(token_ids, slots, strict=True):
            self.k_cache[slot] = self._vector(token_id, 0.17)
            self.v_cache[slot] = self._vector(token_id, 0.31)

    def _online_attention(self, query: list[float], slots: list[int]) -> list[float]:
        row_max = -math.inf
        row_sum = 0.0
        output = [0.0] * self.head_dim
        scale = 1.0 / math.sqrt(self.head_dim)
        for slot in slots:
            key = self.k_cache[slot]
            value = self.v_cache[slot]
            score = sum(q * k for q, k in zip(query, key, strict=True)) * scale
            new_max = max(row_max, score)
            alpha = 0.0 if row_max == -math.inf else math.exp(row_max - new_max)
            beta = math.exp(score - new_max)
            output = [alpha * old + beta * new for old, new in zip(output, value, strict=True)]
            row_sum = alpha * row_sum + beta
            row_max = new_max
        return [value / row_sum for value in output]

    def flash_attention_prefill(self, slots: list[int]) -> None:
        if not slots:
            return
        self.last_attention_output = self._online_attention(self.k_cache[slots[-1]], slots)

    def paged_decode_attention(
        self, block_table: list[int], seq_len: int, block_size: int
    ) -> None:
        slots = [
            block_table[position // block_size] * block_size + position % block_size
            for position in range(seq_len)
        ]
        if not slots:
            return
        self.last_attention_output = self._online_attention(self.k_cache[slots[-1]], slots)

    def attention_batch(self, batch: GPUInputBatch) -> None:
        outputs: list[list[float]] = []
        for request_index in range(batch.num_requests):
            start = batch.context_start_loc[request_index]
            end = batch.context_start_loc[request_index + 1]
            slots = batch.context_slot_mapping[start:end]
            if not slots:
                outputs.append([])
                continue
            outputs.append(self._online_attention(self.k_cache[slots[-1]], slots))
        self.last_batch_attention_outputs = outputs
        if outputs:
            self.last_attention_output = outputs[-1]


class CudaKernelRuntime(KernelRuntime):
    """ctypes adapter for the small C ABI in runtime/minivllm_ops.cu."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        head_dim: int = 64,
        device_index: int = 0,
        max_num_seqs: int = 32,
        max_model_len: int = 2048,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        names = ["minivllm_ops.dll", "libminivllm_ops.so", "libminivllm_ops.dylib"]
        configured = os.environ.get("MINIVLLM_OPS_LIBRARY")
        candidates = [Path(configured)] if configured else []
        candidates.extend(root / "bench" / "bin" / name for name in names)
        library_path = next((path for path in candidates if path.is_file()), None)
        if library_path is None:
            raise RuntimeError(
                "CUDA backend requested, but the miniVLLM operator library is missing; "
                "build it with `make native` or set MINIVLLM_OPS_LIBRARY"
            )
        self._lib = ctypes.CDLL(str(library_path))
        self._lib.mvllm_create.restype = ctypes.c_void_p
        self._lib.mvllm_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.mvllm_destroy.argtypes = [ctypes.c_void_p]
        self._lib.mvllm_append.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._lib.mvllm_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._lib.mvllm_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.mvllm_attention_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self._lib.mvllm_execute_decode_graph.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._handle = self._lib.mvllm_create(
            device_index,
            num_blocks,
            block_size,
            head_dim,
            max_num_seqs,
            max_model_len,
        )
        if not self._handle:
            raise RuntimeError("failed to initialize CUDA KV cache")

    @staticmethod
    def _array(values: list[int]) -> ctypes.Array[ctypes.c_int]:
        return (ctypes.c_int * len(values))(*values)

    def _check(self, status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"CUDA operator {operation} failed with status {status}")

    def append_kv(self, token_ids: list[int], slots: list[int]) -> None:
        self._check(
            self._lib.mvllm_append(
                self._handle, self._array(token_ids), self._array(slots), len(token_ids)
            ),
            "append",
        )

    def flash_attention_prefill(self, slots: list[int]) -> None:
        self._check(
            self._lib.mvllm_prefill(self._handle, self._array(slots), len(slots)),
            "prefill",
        )

    def paged_decode_attention(
        self, block_table: list[int], seq_len: int, block_size: int
    ) -> None:
        self._check(
            self._lib.mvllm_decode(
                self._handle, self._array(block_table), len(block_table), seq_len
            ),
            "decode",
        )

    def attention_batch(self, batch: GPUInputBatch) -> None:
        flat_tables = [block for table in batch.block_tables for block in table]
        max_blocks = len(batch.block_tables[0])
        self._check(
            self._lib.mvllm_attention_batch(
                self._handle,
                self._array(batch.context_slot_mapping),
                self._array(batch.context_start_loc),
                self._array(flat_tables),
                max_blocks,
                self._array(batch.seq_lens),
                self._array([int(value) for value in batch.is_decode]),
                batch.num_requests,
            ),
            "attention batch",
        )

    def execute_decode_graph(self, batch: GPUInputBatch, capture_size: int) -> None:
        if not batch.descriptor.uniform_decode:
            raise ValueError("CUDA Graph execution requires a uniform decode batch")
        flat_tables = [block for table in batch.block_tables for block in table]
        self._check(
            self._lib.mvllm_execute_decode_graph(
                self._handle,
                self._array(batch.input_ids),
                self._array(batch.slot_mapping),
                self._array(flat_tables),
                len(batch.block_tables[0]),
                self._array(batch.seq_lens),
                batch.num_requests,
                capture_size,
            ),
            "decode graph",
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.mvllm_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


def create_kernel_runtime(
    backend: str,
    num_blocks: int,
    block_size: int,
    device_index: int = 0,
    max_num_seqs: int = 32,
    max_model_len: int = 2048,
) -> KernelRuntime:
    if backend == "cuda":
        return CudaKernelRuntime(
            num_blocks,
            block_size,
            device_index=device_index,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
        )
    return ReferenceKernelRuntime()
