import unittest
import types
from unittest.mock import patch
try:
    import torch
except ImportError:  # pragma: no cover
    from typing import Any, cast
    torch = cast(Any, None)
from distributed_kv_manager.engine import init_engine, destroy_engine, should_store, should_retrieve, store_kv, retrieve_kv, StoreStatus, RetrieveStatus

class DummyAttnMeta:
    def __init__(self, seq_lens, slot_mapping):
        self.seq_lens = seq_lens
        self.slot_mapping = slot_mapping

class DummyModelInput:
    def __init__(self, input_tokens, attn_metadata, session_id=b"test_session", layer_id=0):
        self.input_tokens = input_tokens
        self.attn_metadata = attn_metadata
        self.session_id = session_id
        self.layer_id = layer_id

class DummyModelExecutable:
    class DummyEmbed:
        def __init__(self, dim):
            self.embedding_dim = dim
    def __init__(self, hidden_dim=16):
        self.model = types.SimpleNamespace(embed_tokens=DummyModelExecutable.DummyEmbed(hidden_dim))

@unittest.skipIf(torch is None, "torch 未安装，跳过引擎相关测试")
class EngineBasicTests(unittest.TestCase):
    def setUp(self):
        # 构造最小配置: kv_transfer_config 覆盖必要字段
        cfg = types.SimpleNamespace(
            rank=0,
            local_rank=0,
            kv_transfer_config=types.SimpleNamespace(
                storage_type="local",
                local_dir="./tmp_local_storage",
                storage_dir="./tmp_local_storage",
                etcd_endpoints=["127.0.0.1:2379"],
                kv_expire_time=5,
                cleanup_interval=1,
                enable_ssd_caching=False,
                enable_prefetch=False,
            )
        )
        # Patch KVMetadataManager to a dummy so init_engine 不依赖真实 etcd
        class _DummyMM:
            def __init__(self, *a, **kw):
                pass
            def get_metadata(self, key):
                return None
            def get_metadata_by_full_key(self, key):
                return None
            def scan_all_metadata_keys(self):
                return []
            def delete_metadata(self, key):
                return None
            def put_metadata(self, key, meta, replicate=True):
                return None
        with patch("distributed_kv_manager.engine.kv_engine.KVMetadataManager", _DummyMM):
            self.engine = init_engine(cfg)

    def tearDown(self):
        destroy_engine()

    def _make_kv_caches(self, num_layers=2, tokens=4, heads=2, head_dim=4):
        # shape assumption: [2, total_slots, heads, head_dim]
        total_slots = tokens
        kv_caches = []
        for _ in range(num_layers):
            k = torch.randn(total_slots, heads, head_dim)
            v = torch.randn(total_slots, heads, head_dim)
            kv = torch.stack([k, v], dim=0)
            kv_caches.append(kv)
        return kv_caches

    def test_should_store_always(self):
        tokens = torch.randint(0, 100, (4,))
        attn_meta = DummyAttnMeta(seq_lens=[4], slot_mapping=torch.arange(4).unsqueeze(0))
        model_input = DummyModelInput(tokens, attn_meta)
        status = should_store(model_input)
        self.assertEqual(status, StoreStatus.STORED)

    def test_store_and_retrieve_hit(self):
        tokens = torch.randint(0, 50, (4,))
        attn_meta = DummyAttnMeta(seq_lens=[4], slot_mapping=torch.arange(4).unsqueeze(0))
        model_input = DummyModelInput(tokens, attn_meta)
        kv_caches = self._make_kv_caches()
        model_exec = DummyModelExecutable(hidden_dim=8)

        # store
        store_kv(None, None, None, model_exec, model_input, kv_caches, StoreStatus.STORED)
        # 等待异步写入任务完成（避免重新初始化触发真实etcd连接）
        for f in getattr(self.engine, "_futures", []):
            try:
                f.result(timeout=15)
            except Exception:
                pass

        rstatus = should_retrieve(model_input)
        self.assertEqual(rstatus, RetrieveStatus.HIT)

    def test_retrieve_miss_before_store(self):
        tokens = torch.randint(0, 50, (4,))
        attn_meta = DummyAttnMeta(seq_lens=[4], slot_mapping=torch.arange(4).unsqueeze(0))
        model_input = DummyModelInput(tokens, attn_meta)
        rstatus = should_retrieve(model_input)
        self.assertEqual(rstatus, RetrieveStatus.MISS)

if __name__ == "__main__":
    unittest.main()
