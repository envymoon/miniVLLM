from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import uuid

from ..config import EngineConfig, SamplingParams
from ..protocol import CommandType, EngineCommand, EngineRequest, RequestOutput
from ..tokenizer import CharacterTokenizer
from .core import engine_core_process_main
from .dp_coordinator import dp_coordinator_process_main


class EngineCoreClient:
    """Multiprocess transport shared by the sync and async public engines."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.tokenizer = CharacterTokenizer()
        self.context = mp.get_context(config.process_start_method)
        self.command_queue = self.context.Queue()
        self.output_queue = self.context.Queue()
        self._streams: dict[str, queue.Queue[RequestOutput]] = {}
        self._streams_lock = threading.Lock()
        self._closed = False
        process_target = (
            dp_coordinator_process_main
            if config.data_parallel_size > 1
            else engine_core_process_main
        )
        process_name = (
            "minivllm-dp-coordinator"
            if config.data_parallel_size > 1
            else "minivllm-engine-core"
        )
        self.process = self.context.Process(
            target=process_target,
            args=(config, self.command_queue, self.output_queue),
            name=process_name,
        )
        self.process.start()
        self._output_thread = threading.Thread(
            target=self._route_outputs, name="minivllm-output-router", daemon=True
        )
        self._output_thread.start()

    def _route_outputs(self) -> None:
        while True:
            output: RequestOutput = self.output_queue.get()
            with self._streams_lock:
                stream = self._streams.get(output.request_id)
            if stream is not None:
                stream.put(output)

    def submit(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> tuple[str, queue.Queue[RequestOutput]]:
        if self._closed:
            raise RuntimeError("engine is closed")
        request_id = request_id or uuid.uuid4().hex
        stream: queue.Queue[RequestOutput] = queue.Queue()
        with self._streams_lock:
            if request_id in self._streams:
                raise ValueError(f"duplicate request id: {request_id}")
            self._streams[request_id] = stream
        prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids:
            prompt_token_ids = [self.tokenizer.bos_token_id]
        request = EngineRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        self.command_queue.put(EngineCommand(CommandType.ADD, request=request))
        return request_id, stream

    def finish_stream(self, request_id: str) -> None:
        with self._streams_lock:
            self._streams.pop(request_id, None)

    def abort(self, request_id: str) -> None:
        if not self._closed:
            self.command_queue.put(
                EngineCommand(CommandType.ABORT, request_id=request_id)
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.command_queue.put(EngineCommand(CommandType.SHUTDOWN))
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.command_queue.close()
        self.output_queue.close()

    def __enter__(self) -> EngineCoreClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
