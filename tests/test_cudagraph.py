import unittest

from minivllm.protocol import WorkerBatch, WorkerItem
from minivllm.worker.batch import BatchDescriptor, GPUInputBatch
from minivllm.worker.cudagraph import CUDAGraphDispatcher, CUDAGraphManager


class GraphRunner:
    def __init__(self) -> None:
        self.capture_size = None

    def execute_model(self, batch: GPUInputBatch) -> dict[str, int]:
        return {"mode": 0}

    def execute_cudagraph(
        self, batch: GPUInputBatch, capture_size: int
    ) -> dict[str, int]:
        self.capture_size = capture_size
        return {"mode": 1}


class CUDAGraphDispatcherTest(unittest.TestCase):
    def test_decode_batch_uses_next_capture_bucket(self) -> None:
        dispatcher = CUDAGraphDispatcher("full_decode_only", (1, 2, 4, 8))

        decision = dispatcher.dispatch(BatchDescriptor(3, 3, True))

        self.assertTrue(decision.wants_graph)
        self.assertEqual(decision.capture_size, 4)

    def test_mixed_and_oversized_batches_fall_back_to_eager(self) -> None:
        dispatcher = CUDAGraphDispatcher("full_decode_only", (1, 2, 4))

        mixed = dispatcher.dispatch(BatchDescriptor(4, 2, False))
        oversized = dispatcher.dispatch(BatchDescriptor(8, 8, True))

        self.assertFalse(mixed.wants_graph)
        self.assertFalse(oversized.wants_graph)

    def test_manager_invokes_graph_capable_runner(self) -> None:
        batch = GPUInputBatch.from_worker_batch(
            WorkerBatch(
                1,
                [WorkerItem("r", [3], 1, [1], [0], 2, 1, True)],
            ),
            block_size=4,
        )
        runner = GraphRunner()
        manager = CUDAGraphManager("full_decode_only", (1, 2, 4))

        output, mode = manager.execute(runner, batch)

        self.assertEqual(output, {"mode": 1})
        self.assertEqual(mode, "cuda_graph")
        self.assertEqual(runner.capture_size, 1)


if __name__ == "__main__":
    unittest.main()
