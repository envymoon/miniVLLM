import unittest

from minivllm.worker.operator_runtime import ReferenceKernelRuntime


class OperatorRuntimeTest(unittest.TestCase):
    def test_prefill_and_paged_decode_share_physical_cache(self) -> None:
        runtime = ReferenceKernelRuntime(head_dim=4)
        runtime.append_kv([10, 20, 30], [4, 5, 0])
        runtime.flash_attention_prefill([4, 5])
        prefill_output = runtime.last_attention_output
        runtime.paged_decode_attention(
            block_table=[2, 0], seq_len=3, block_size=2
        )

        self.assertEqual(len(prefill_output), 4)
        self.assertEqual(len(runtime.last_attention_output), 4)
        self.assertNotEqual(prefill_output, runtime.last_attention_output)


if __name__ == "__main__":
    unittest.main()
