import unittest
import time
from distributed_kv_manager.prefetch import PrefetchBuffer, EntryState

class PrefetchBufferTests(unittest.TestCase):
    def test_reserve_and_ready(self):
        buf = PrefetchBuffer(capacity=2)
        self.assertTrue(buf.reserve("k1"))
        self.assertFalse(buf.reserve("k1"))  # duplicate
        buf.mark_fetching("k1")
        buf.mark_ready("k1")
        self.assertTrue(buf.is_ready("k1"))

    def test_eviction(self):
        buf = PrefetchBuffer(capacity=1)
        buf.reserve("k1")
        buf.reserve("k2")  # should evict k1 by simple FIFO
        self.assertFalse(buf.is_ready("k1"))
        buf.mark_ready("k2")
        self.assertTrue(buf.is_ready("k2"))

if __name__ == "__main__":
    unittest.main()
