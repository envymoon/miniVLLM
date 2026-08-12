import math
import unittest

try:
    import torch
except ImportError:
    torch = None

from minivllm.protocol import WorkerBatch, WorkerItem
from minivllm.worker.batch import GPUInputBatch


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionBackendTest(unittest.TestCase):
    def test_backend_selection_is_explicit(self) -> None:
        from minivllm.model_executor.attention import (
            TorchAttentionBackend,
            create_attention_backend,
        )

        backend = create_attention_backend("auto", torch.device("cpu"))

        self.assertIsInstance(backend, TorchAttentionBackend)
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            create_attention_backend("custom", torch.device("cpu"))

    def test_mixed_attention_matches_independent_paged_reference(self) -> None:
        from minivllm.model_executor.attention import (
            AttentionMetadata,
            TorchAttentionBackend,
        )

        torch.manual_seed(7)
        query_heads = 4
        kv_heads = 2
        head_dim = 4
        key_cache = torch.zeros(12, kv_heads, head_dim)
        value_cache = torch.zeros_like(key_cache)
        key_cache[4:6] = torch.randn(2, kv_heads, head_dim)
        value_cache[4:6] = torch.randn(2, kv_heads, head_dim)
        batch = GPUInputBatch.from_worker_batch(
            WorkerBatch(
                2,
                [
                    WorkerItem("decode", [9], 2, [6], [1], 3, 1, True),
                    WorkerItem("prefill", [10, 11], 0, [0, 1], [0], 2, 0, True),
                ],
            ),
            block_size=4,
        )
        metadata = AttentionMetadata.from_batch(
            batch, torch.device("cpu"), block_size=4
        )
        query = torch.randn(3, query_heads, head_dim)
        key = torch.randn(3, kv_heads, head_dim)
        value = torch.randn(3, kv_heads, head_dim)
        expected_key_cache = key_cache.clone()
        expected_value_cache = value_cache.clone()
        expected_key_cache[torch.tensor(batch.slot_mapping)] = key
        expected_value_cache[torch.tensor(batch.slot_mapping)] = value

        output = TorchAttentionBackend().forward(
            query, key, value, key_cache, value_cache, metadata
        )

        expected = torch.zeros_like(output)
        group_size = query_heads // kv_heads
        for request in range(batch.num_requests):
            start = batch.query_start_loc[request]
            end = batch.query_start_loc[request + 1]
            for query_index in range(start, end):
                position = batch.positions[query_index]
                slots = [
                    batch.block_tables[request][token // 4] * 4 + token % 4
                    for token in range(position + 1)
                ]
                for query_head in range(query_heads):
                    kv_head = query_head // group_size
                    keys = expected_key_cache[slots, kv_head]
                    values = expected_value_cache[slots, kv_head]
                    scores = keys @ query[query_index, query_head] / math.sqrt(head_dim)
                    expected[query_index, query_head] = (
                        torch.softmax(scores, dim=0).unsqueeze(0) @ values
                    ).squeeze(0)

        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(key_cache, expected_key_cache)
        torch.testing.assert_close(value_cache, expected_value_cache)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA is unavailable"
    )
    def test_custom_cuda_matches_torch_backend(self) -> None:
        from minivllm.model_executor.attention import (
            AttentionMetadata,
            CudaAttentionBackend,
            TorchAttentionBackend,
        )

        torch.manual_seed(11)
        device = torch.device("cuda")
        batch = GPUInputBatch.from_worker_batch(
            WorkerBatch(
                3,
                [
                    WorkerItem("decode", [9], 2, [6], [1], 3, 1, True),
                    WorkerItem("prefill", [10, 11], 0, [0, 1], [0], 2, 0, True),
                ],
            ),
            block_size=4,
        )
        metadata = AttentionMetadata.from_batch(batch, device, block_size=4)
        query = torch.randn(3, 4, 4, device=device)
        key = torch.randn(3, 2, 4, device=device)
        value = torch.randn(3, 2, 4, device=device)
        initial_key = torch.zeros(12, 2, 4, device=device)
        initial_value = torch.zeros_like(initial_key)
        initial_key[4:6] = torch.randn(2, 2, 4, device=device)
        initial_value[4:6] = torch.randn(2, 2, 4, device=device)
        reference_key = initial_key.clone()
        reference_value = initial_value.clone()
        custom_key = initial_key.clone()
        custom_value = initial_value.clone()

        reference = TorchAttentionBackend().forward(
            query, key, value, reference_key, reference_value, metadata
        )
        custom = CudaAttentionBackend().forward(
            query, key, value, custom_key, custom_value, metadata
        )

        torch.testing.assert_close(custom, reference, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(custom_key, reference_key)
        torch.testing.assert_close(custom_value, reference_value)


if __name__ == "__main__":
    unittest.main()
