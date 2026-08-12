from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..config import EngineConfig, SamplingParams
from ..protocol import EngineMetricsSnapshot, RequestOutput
from .client import EngineCoreClient


class AsyncLLMEngine:
    """Async streaming facade; scheduling remains inside EngineCore."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.client = EngineCoreClient(self.config)

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[RequestOutput]:
        request_id, stream = self.client.submit(
            prompt, sampling_params or SamplingParams(), request_id
        )
        finished = False
        try:
            while not finished:
                output = await asyncio.to_thread(stream.get)
                finished = output.finished
                yield output
        finally:
            if not finished:
                self.client.abort(request_id)
            self.client.finish_stream(request_id)

    async def abort(self, request_id: str) -> None:
        self.client.abort(request_id)

    @property
    def metrics(self) -> EngineMetricsSnapshot:
        return self.client.metrics

    def close(self) -> None:
        self.client.close()

    async def __aenter__(self) -> AsyncLLMEngine:
        return self

    async def __aexit__(self, *_: object) -> None:
        self.close()
