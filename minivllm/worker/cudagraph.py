from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .batch import BatchDescriptor, GPUInputBatch


class GraphCapableModelRunner(Protocol):
    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]: ...

    def execute_cudagraph(
        self, batch: GPUInputBatch, capture_size: int
    ) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class CUDAGraphDecision:
    requested_mode: str
    capture_size: int | None

    @property
    def wants_graph(self) -> bool:
        return self.capture_size is not None


class CUDAGraphDispatcher:
    """Central source of truth for graph-eligible execution shapes."""

    def __init__(self, mode: str, capture_sizes: tuple[int, ...]) -> None:
        self.mode = mode
        self.capture_sizes = capture_sizes

    def dispatch(self, descriptor: BatchDescriptor) -> CUDAGraphDecision:
        if self.mode == "none" or not descriptor.uniform_decode:
            return CUDAGraphDecision("none", None)
        required_size = max(descriptor.num_tokens, descriptor.num_requests)
        capture_size = next(
            (size for size in self.capture_sizes if size >= required_size), None
        )
        if capture_size is None:
            return CUDAGraphDecision("none", None)
        return CUDAGraphDecision("full_decode_only", capture_size)


class CUDAGraphManager:
    """Dispatches graph-safe batches and preserves an explicit eager fallback."""

    def __init__(self, mode: str, capture_sizes: tuple[int, ...]) -> None:
        self.dispatcher = CUDAGraphDispatcher(mode, capture_sizes)

    def execute(
        self, runner: GraphCapableModelRunner, batch: GPUInputBatch
    ) -> tuple[dict[str, int], str]:
        decision = self.dispatcher.dispatch(batch.descriptor)
        execute_graph = getattr(runner, "execute_cudagraph", None)
        if decision.wants_graph and execute_graph is not None:
            return execute_graph(batch, decision.capture_size), "cuda_graph"
        return runner.execute_model(batch), "eager"
