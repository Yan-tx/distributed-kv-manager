import unittest
import os
import shutil
try:
    import torch
except ImportError:  # pragma: no cover
    from typing import Any, cast
    torch = cast(Any, None)
from distributed_kv_manager.storage.local_storage import LocalStorage

class LocalStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = "./tmp_local_storage_case"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.storage = LocalStorage(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @unittest.skipIf(torch is None, "torch 未安装，跳过张量读写测试")
    def test_upload_download_roundtrip(self):
        k = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 4)
        input_tokens = torch.randint(0, 100, (3,))
        roi = torch.ones_like(input_tokens, dtype=torch.bool)
        payload = self.storage.pack_kv_data(k, v, input_tokens, roi)
        ok = self.storage.upload("file1.pt", payload)
        self.assertTrue(ok)
        data = self.storage.download("file1.pt")
        self.assertIsNotNone(data)
        k2, v2 = self.storage.unpack_kv_data(data)  # type: ignore[arg-type]
        self.assertTrue(torch.allclose(k, k2))
        self.assertTrue(torch.allclose(v, v2))

    def test_exists_delete(self):
        data = b"abc"
        self.storage.upload("filex.bin", data)
        self.assertTrue(self.storage.exists("filex.bin"))
        self.storage.delete("filex.bin")
        self.assertFalse(self.storage.exists("filex.bin"))

if __name__ == "__main__":
    unittest.main()
