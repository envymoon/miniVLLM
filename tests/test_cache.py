import unittest

from minivllm.cache import KVCacheManager


class KVCacheManagerTest(unittest.TestCase):
    def test_slot_mapping_and_release(self) -> None:
        manager = KVCacheManager(num_blocks=3, block_size=4)
        manager.create("r1")
        manager.reserve("r1", 5)

        self.assertEqual(manager.block_table("r1"), [0, 1])
        self.assertEqual(manager.slots("r1", 2, 3), [2, 3, 4])
        self.assertEqual(manager.pool.num_free_blocks, 1)

        manager.release("r1")
        self.assertEqual(manager.pool.num_free_blocks, 3)


if __name__ == "__main__":
    unittest.main()
