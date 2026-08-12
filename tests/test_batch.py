import unittest

from minivllm.protocol import WorkerBatch, WorkerItem
from minivllm.worker.batch import GPUInputBatch
from minivllm.worker.model_runner import BootstrapModelRunner
from minivllm.worker.operator_runtime import ReferenceKernelRuntime


class CountingRuntime(ReferenceKernelRuntime):
    def __init__(self) -> None:
        super().__init__(head_dim=4)
        self.append_calls = 0
        self.batch_attention_calls = 0

    def append_kv(self, token_ids: list[int], slots: list[int]) -> None:
        self.append_calls += 1
        super().append_kv(token_ids, slots)

    def attention_batch(self, batch: GPUInputBatch) -> None:
        self.batch_attention_calls += 1
        super().attention_batch(batch)


class GPUInputBatchTest(unittest.TestCase):
    def _mixed_batch(self) -> GPUInputBatch:
        return GPUInputBatch.from_worker_batch(
            WorkerBatch(
                batch_id=7,
                items=[
                    WorkerItem(
                        request_id="decode",
                        token_ids=[13],
                        start_pos=2,
                        slot_mapping=[2],
                        block_table=[0],
                        seq_len=3,
                        generated_count=1,
                        should_sample=True,
                    ),
                    WorkerItem(
                        request_id="prefill",
                        token_ids=[20, 21],
                        start_pos=0,
                        slot_mapping=[4, 5],
                        block_table=[1],
                        seq_len=2,
                        generated_count=0,
                        should_sample=True,
                    ),
                ],
            ),
            block_size=4,
        )

    def test_flattens_mixed_prefill_decode_metadata(self) -> None:
        batch = self._mixed_batch()

        self.assertEqual(batch.input_ids, [13, 20, 21])
        self.assertEqual(batch.positions, [2, 0, 1])
        self.assertEqual(batch.query_start_loc, [0, 1, 3])
        self.assertEqual(batch.context_start_loc, [0, 3, 5])
        self.assertEqual(batch.is_decode, [True, False])
        self.assertEqual(batch.num_decode_tokens, 1)
        self.assertEqual(batch.num_prefill_tokens, 2)
        self.assertFalse(batch.descriptor.uniform_decode)

    def test_bootstrap_runner_executes_one_batch_call(self) -> None:
        runtime = CountingRuntime()
        runner = BootstrapModelRunner(runtime, block_size=4)
        batch = self._mixed_batch()
        runtime.append_kv([11, 12], [0, 1])
        runtime.append_calls = 0

        sampled = runner.execute_model(batch)

        self.assertEqual(set(sampled), {"decode", "prefill"})
        self.assertEqual(runtime.append_calls, 1)
        self.assertEqual(runtime.batch_attention_calls, 1)
        self.assertEqual(len(runtime.last_batch_attention_outputs), 2)


if __name__ == "__main__":
    unittest.main()
