from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..config import EngineConfig, SamplingParams
from ..engine.async_engine import AsyncLLMEngine


def _handler(engine: AsyncLLMEngine) -> type[BaseHTTPRequestHandler]:
    class CompletionHandler(BaseHTTPRequestHandler):
        server_version = "miniVLLM/0.1"

        def _json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/metrics":
                metrics = asdict(engine.metrics)
                metrics["average_tokens_per_batch"] = (
                    engine.metrics.average_tokens_per_batch
                )
                self._json(HTTPStatus.OK, metrics)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/completions":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                prompt = body["prompt"]
                stop_value = body.get("stop", ())
                if isinstance(stop_value, str):
                    stop_value = (stop_value,)
                params = SamplingParams(
                    max_tokens=int(body.get("max_tokens", 16)),
                    temperature=float(body.get("temperature", 0.0)),
                    stop=tuple(stop_value),
                )
                if body.get("stream", False):
                    self._stream(prompt, params)
                else:
                    self._complete(prompt, params)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _complete(self, prompt: str, params: SamplingParams) -> None:
            async def collect() -> tuple[str, str | None, dict[str, Any] | None]:
                text = ""
                reason = None
                request_metrics = None
                async for output in engine.generate(prompt, params):
                    if output.error:
                        raise RuntimeError(output.error)
                    text += output.text
                    reason = output.finish_reason.value if output.finish_reason else None
                    if output.request_metrics is not None:
                        request_metrics = asdict(output.request_metrics)
                return text, reason, request_metrics

            try:
                text, reason, request_metrics = asyncio.run(collect())
                self._json(
                    HTTPStatus.OK,
                    {
                        "object": "text_completion",
                        "choices": [{"text": text, "finish_reason": reason, "index": 0}],
                        "metrics": request_metrics,
                    },
                )
            except RuntimeError as error:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def _stream(self, prompt: str, params: SamplingParams) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            async def write_events() -> None:
                async for output in engine.generate(prompt, params):
                    data = {
                        "choices": [
                            {
                                "text": output.text,
                                "finish_reason": (
                                    output.finish_reason.value
                                    if output.finish_reason
                                    else None
                                ),
                                "index": 0,
                            }
                        ]
                    }
                    self.wfile.write(
                        f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()
                    )
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")

            try:
                asyncio.run(write_events())
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, *_: object) -> None:
            return

    return CompletionHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the miniVLLM API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    parser.add_argument("--backend", choices=("reference", "cuda"), default="reference")
    parser.add_argument("--model", default=None, help="local Hugging Face Llama directory")
    parser.add_argument("--tokenizer", default=None, help="optional local tokenizer directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("none", "full_decode_only"),
        default="none",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EngineConfig(
        data_parallel_size=args.data_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_gpu_blocks=args.num_gpu_blocks,
        worker_backend=args.backend,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        cudagraph_mode=args.cudagraph_mode,
    )
    engine = AsyncLLMEngine(config)
    server = ThreadingHTTPServer((args.host, args.port), _handler(engine))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
