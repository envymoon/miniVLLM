import asyncio
import unittest

from minivllm import AsyncLLMEngine, EngineConfig, LLMEngine, SamplingParams
from minivllm.protocol import FinishReason


class EngineProcessTest(unittest.TestCase):
    def test_sync_engine_streams_and_finishes(self) -> None:
        config = EngineConfig(
            data_parallel_size=2,
            max_num_batched_tokens=8,
            block_size=4,
            num_gpu_blocks=8,
        )
        with LLMEngine(config) as engine:
            outputs = list(engine.generate("hello", SamplingParams(max_tokens=4)))

        self.assertEqual("".join(item.text for item in outputs), " min")
        self.assertTrue(outputs[-1].finished)
        self.assertEqual(outputs[-1].finish_reason, FinishReason.LENGTH)

    def test_async_engine_handles_concurrent_requests(self) -> None:
        async def run() -> list[str]:
            engine = AsyncLLMEngine(
                EngineConfig(
                    data_parallel_size=2,
                    max_num_batched_tokens=16,
                    block_size=4,
                    num_gpu_blocks=8,
                )
            )

            async def collect(prompt: str) -> str:
                text = ""
                async for output in engine.generate(
                    prompt, SamplingParams(max_tokens=3)
                ):
                    text += output.text
                return text

            try:
                return await asyncio.gather(collect("a"), collect("b"))
            finally:
                engine.close()

        self.assertEqual(asyncio.run(run()), [" mi", " mi"])


if __name__ == "__main__":
    unittest.main()
