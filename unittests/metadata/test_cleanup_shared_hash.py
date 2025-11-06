import unittest
import types
import time
from distributed_kv_manager.metadata.cleanup import KVCleanupManager
from distributed_kv_manager.metadata.etcd import KVMetadata

class DummyMetaManager:
    def __init__(self):
        self._store = {}
        self.scans = 0
    def scan_all_metadata_keys(self):
        self.scans += 1
        return list(self._store.keys())
    def get_metadata_by_full_key(self, key):
        return self._store.get(key)
    def delete_metadata(self, file_path):
        self._store.pop(file_path, None)

class DummyStorage:
    def __init__(self):
        self.deleted = []
    def delete(self, file_path):
        self.deleted.append(file_path)
        return True

class CleanupSharedHashTests(unittest.TestCase):
    def _make_meta(self, file_path, expired=False):
        now = int(time.time())
        expire = now - 1 if expired else now + 1000
        return KVMetadata(
            session_id=b"s", layer_id=0, token_idx="0-1", file_path=file_path, file_size=0,
            create_time=now, last_access=now, expire_time=expire,
            replica_locations=[b"" for _ in range(3)], status=1, schema_version=1,
            ext_flags=0, ext_data=b"", ext_data_len=0
        )

    def test_skip_delete_when_other_reference(self):
        mm = DummyMetaManager()
        st = DummyStorage()
        # 两个文件不同前缀同hash
        h = "abcd1234ef567890"
        f1 = f"kv_a_layer_0_{h}.pt"
        f2 = f"kv_b_layer_0_{h}.pt"
        mm._store[f1] = self._make_meta(f1, expired=True)
        mm._store[f2] = self._make_meta(f2, expired=False)
        mgr = KVCleanupManager(mm, cleanup_interval=1, storage=st)  # type: ignore[arg-type]
        # 直接调用内部方法
        mgr._cleanup_expired_kv(mm._store[f1])
        # f1 metadata 应被删除但物理文件不删除（因为f2引用仍有效）
        self.assertNotIn(f1, mm._store)
        self.assertEqual(st.deleted, [])

    def test_physical_delete_when_single(self):
        mm = DummyMetaManager()
        st = DummyStorage()
        h = "abcd1234ef567890"
        f1 = f"kv_a_layer_0_{h}.pt"
        mm._store[f1] = self._make_meta(f1, expired=True)
        mgr = KVCleanupManager(mm, cleanup_interval=1, storage=st)  # type: ignore[arg-type]
        mgr._cleanup_expired_kv(mm._store[f1])
        self.assertNotIn(f1, mm._store)
        self.assertIn(f1, st.deleted)

if __name__ == "__main__":
    unittest.main()
