from __future__ import annotations

from collections.abc import Iterator

from ..config import EngineConfig, SamplingParams
from ..protocol import EngineMetricsSnapshot, RequestOutput
from .client import EngineCoreClient


class LLMEngine:
    """Synchronous public engine backed by a separate EngineCore process."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.client = EngineCoreClient(self.config)

    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Iterator[RequestOutput]:
        request_id, stream = self.client.submit(
            prompt, sampling_params or SamplingParams(), request_id
        )
        finished = False
        try:
            while not finished:
                output = stream.get()
                finished = output.finished
                yield output
        finally:
            if not finished:
                self.client.abort(request_id)
            self.client.finish_stream(request_id)

    def abort(self, request_id: str) -> None:
        self.client.abort(request_id)

    @property
    def metrics(self) -> EngineMetricsSnapshot:
        return self.client.metrics

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
