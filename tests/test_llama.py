import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None
    save_file = None

from minivllm.protocol import WorkerBatch, WorkerItem
from minivllm.worker.batch import GPUInputBatch


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MinimalLlamaTest(unittest.TestCase):
    def test_mixed_batch_uses_paged_layer_cache(self) -> None:
        from minivllm.model_executor.attention import AttentionMetadata
        from minivllm.model_executor.llama import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        model = LlamaForCausalLM(config).eval()
        cache_shape = (16, config.num_key_value_heads, config.head_dim)
        key_cache = [torch.empty(cache_shape) for _ in range(config.num_hidden_layers)]
        value_cache = [torch.empty(cache_shape) for _ in range(config.num_hidden_layers)]

        prefill = GPUInputBatch.from_worker_batch(
            WorkerBatch(
                1,
                [
                    WorkerItem(
                        "r0", [2, 3], 0, [0, 1], [0], 2, 0, True
                    )
                ],
            ),
            block_size=4,
        )
        with torch.inference_mode():
            first_logits = model(
                torch.tensor(prefill.input_ids),
                torch.tensor(prefill.positions),
                AttentionMetadata.from_batch(
                    prefill, torch.device("cpu"), block_size=4
                ),
                key_cache,
                value_cache,
            )

        mixed = GPUInputBatch.from_worker_batch(
            WorkerBatch(
                2,
                [
                    WorkerItem("r0", [4], 2, [2], [0], 3, 1, True),
                    WorkerItem("r1", [6, 7], 0, [4, 5], [1], 2, 0, True),
                ],
            ),
            block_size=4,
        )
        with torch.inference_mode():
            mixed_logits = model(
                torch.tensor(mixed.input_ids),
                torch.tensor(mixed.positions),
                AttentionMetadata.from_batch(
                    mixed, torch.device("cpu"), block_size=4
                ),
                key_cache,
                value_cache,
            )

        self.assertEqual(tuple(first_logits.shape), (2, config.vocab_size))
        self.assertEqual(tuple(mixed_logits.shape), (3, config.vocab_size))
        self.assertTrue(torch.isfinite(mixed_logits).all())

    def test_local_safetensors_runner_loads_and_samples(self) -> None:
        from minivllm.model_executor.llama import (
            LlamaConfig,
            LlamaForCausalLM,
            LlamaModelRunner,
        )

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "intermediate_size": config.intermediate_size,
                        "num_hidden_layers": config.num_hidden_layers,
                        "num_attention_heads": config.num_attention_heads,
                        "num_key_value_heads": config.num_key_value_heads,
                        "max_position_embeddings": config.max_position_embeddings,
                    }
                ),
                encoding="utf-8",
            )
            save_file(model.state_dict(), root / "model.safetensors")
            runner = LlamaModelRunner(
                str(root),
                num_gpu_blocks=4,
                block_size=4,
                max_model_len=16,
                device="cpu",
                dtype="float32",
                data_parallel_rank=0,
            )
            batch = GPUInputBatch.from_worker_batch(
                WorkerBatch(
                    1,
                    [WorkerItem("r", [2, 3], 0, [0, 1], [0], 2, 0, True)],
                ),
                block_size=4,
            )

            sampled = runner.execute_model(batch)

            self.assertIn(sampled["r"], range(config.vocab_size))
            self.assertFalse(runner.supports_cudagraph)
            decode_batch = GPUInputBatch.from_worker_batch(
                WorkerBatch(
                    2,
                    [WorkerItem("r", [4], 1, [1], [0], 2, 1, True)],
                ),
                block_size=4,
            )
            state = runner._create_graph_state(4)
            runner._copy_decode_batch(state, decode_batch)
            self.assertEqual(
                state.metadata.active_mask.tolist(), [True, False, False, False]
            )
            self.assertEqual(state.metadata.slot_mapping[0].item(), 1)
            self.assertEqual(
                state.metadata.slot_mapping[1:].tolist(),
                [runner.physical_cache_slots + index for index in range(1, 4)],
            )

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA is unavailable"
    )
    def test_custom_llama_captures_and_replays_decode_graph(self) -> None:
        from minivllm.model_executor.llama import (
            LlamaConfig,
            LlamaForCausalLM,
            LlamaModelRunner,
        )

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        )
        model = LlamaForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "llama",
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "intermediate_size": config.intermediate_size,
                        "num_hidden_layers": config.num_hidden_layers,
                        "num_attention_heads": config.num_attention_heads,
                        "num_key_value_heads": config.num_key_value_heads,
                        "max_position_embeddings": config.max_position_embeddings,
                    }
                ),
                encoding="utf-8",
            )
            save_file(model.state_dict(), root / "model.safetensors")
            runner = LlamaModelRunner(
                str(root),
                num_gpu_blocks=4,
                block_size=4,
                max_model_len=16,
                device="cuda:0",
                dtype="float32",
                data_parallel_rank=0,
                max_num_seqs=2,
                attention_backend="custom",
            )
            prefill = GPUInputBatch.from_worker_batch(
                WorkerBatch(
                    1,
                    [WorkerItem("r", [2, 3], 0, [0, 1], [0], 2, 0, True)],
                ),
                block_size=4,
            )
            first_decode = GPUInputBatch.from_worker_batch(
                WorkerBatch(
                    2,
                    [WorkerItem("r", [4], 2, [2], [0], 3, 1, True)],
                ),
                block_size=4,
            )
            second_decode = GPUInputBatch.from_worker_batch(
                WorkerBatch(
                    3,
                    [WorkerItem("r", [5], 3, [3], [0], 4, 2, True)],
                ),
                block_size=4,
            )

            runner.execute_model(prefill)
            captured = runner.execute_cudagraph(first_decode, capture_size=2)
            replayed = runner.execute_cudagraph(second_decode, capture_size=2)

            self.assertTrue(runner.supports_cudagraph)
            self.assertIn(captured["r"], range(config.vocab_size))
            self.assertIn(replayed["r"], range(config.vocab_size))
            self.assertIsNotNone(runner._graph_states[2].graph)
            runner.close()


if __name__ == "__main__":
    unittest.main()
