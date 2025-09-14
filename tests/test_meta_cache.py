# test.py
import torch
import time
from distributed_kv_manager.engine.kv_engine import (
    init_engine, destroy_engine,
    should_store, store_kv,
    should_retrieve, retrieve_kv,
    StoreStatus, RetrieveStatus
)

# ------------------ Mock Config ------------------ #
class MockKVTransferConfig:
    crail_dir = "./test_crail_kvcache"
    etcd_endpoints = ["127.0.0.1:2379"]

class MockConfig:
    kv_transfer_config = MockKVTransferConfig()
    rank = 0
    local_rank = 0

# ------------------ Mock Model Input ------------------ #
class MockAttnMetadata:
    def __init__(self, seq_lens, slot_mapping, num_prefill_tokens):
        self.seq_lens = seq_lens
        self.slot_mapping = slot_mapping
        self.num_prefill_tokens = num_prefill_tokens

class MockModelInput:
    def __init__(self, input_tokens, attn_metadata):
        self.input_tokens = input_tokens
        self.attn_metadata = attn_metadata
        self.hidden_states = torch.randn(len(input_tokens), 4, 8)  # mock hidden state
        self.layer_id = 0
        self.session_id = b"session_0000"

# ------------------ 测试函数 ------------------ #
def test_engine():
    # 初始化 engine
    config = MockConfig()
    engine = init_engine(config)
    print("[TEST] Engine initialized")

    # 准备模拟数据
    seq_lens = [3, 2]  # 两段序列
    slot_mapping = torch.tensor([0, 1, 2, 0, 1])  # mock slot映射
    input_tokens = torch.tensor([101, 102, 103, 201, 202])
    attn_metadata = MockAttnMetadata(seq_lens, slot_mapping, num_prefill_tokens=5)
    model_input = MockModelInput(input_tokens, attn_metadata)

    # 准备 KV 缓存
    kv_caches = [torch.randn(2, 5, 4) for _ in range(2)]  # mock KV cache 2层

    # ------------------ 测试 should_store ------------------ #
    store_status = should_store(model_input)
    assert store_status == StoreStatus.STORED
    print("[TEST] should_store OK")

    # ------------------ 测试 store_kv ------------------ #
    store_kv(None, None, None, None, model_input, kv_caches, store_status)
    print("[TEST] store_kv OK, waiting async writes...")
    time.sleep(2)  # 等待异步写入完成

    # ------------------ 测试 should_retrieve ------------------ #
    retrieve_status = should_retrieve(model_input)
    print(f"[TEST] should_retrieve: {retrieve_status}")
    assert retrieve_status == RetrieveStatus.HIT

    # ------------------ 测试 retrieve_kv ------------------ #
    final_hidden, hit, _ = retrieve_kv(None, model_input, kv_caches, retrieve_status)
    print(f"[TEST] retrieve_kv hit: {hit}, final_hidden shape: {final_hidden.shape if final_hidden is not None else None}")

    # 清理 engine
    destroy_engine()
    print("[TEST] Engine destroyed")

if __name__ == "__main__":
    test_engine()
    
