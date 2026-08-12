import unittest

from minivllm.config import EngineConfig, SamplingParams
from minivllm.protocol import EngineRequest
from minivllm.scheduler import Scheduler


class SchedulerTest(unittest.TestCase):
    def test_chunked_prefill_uses_global_token_budget(self) -> None:
        scheduler = Scheduler(
            EngineConfig(
                data_parallel_size=2,
                max_num_batched_tokens=4,
                block_size=2,
                num_gpu_blocks=8,
            )
        )
        for index in range(2):
            scheduler.add_request(
                EngineRequest(
                    request_id=f"r{index}",
                    prompt="abc",
                    prompt_token_ids=[10, 11, 12],
                    sampling_params=SamplingParams(max_tokens=2),
                )
            )

        schedule = scheduler.schedule()
        self.assertEqual(set(schedule.by_rank), {0})
        self.assertEqual(sum(schedule.scheduled_counts.values()), 4)
        self.assertEqual(schedule.by_rank[0][0].token_ids, [10, 11, 12])
        self.assertEqual(schedule.by_rank[0][1].token_ids, [10])

    def test_prompt_larger_than_rank_cache_is_rejected(self) -> None:
        scheduler = Scheduler(
            EngineConfig(block_size=2, num_gpu_blocks=1, max_num_batched_tokens=2)
        )
        with self.assertRaisesRegex(ValueError, "KV cache"):
            scheduler.add_request(
                EngineRequest("r", "abc", [10, 11, 12], SamplingParams())
            )


if __name__ == "__main__":
    unittest.main()
