import torch
import os
import sys
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distributed_kv_manager.engine.kv_engine import (
    init_engine, 
    should_store, 
    store_kv, 
    should_retrieve, 
    retrieve_kv,
    StoreStatus,
    RetrieveStatus
)
from distributed_kv_manager.config_loader import load_config_from_json


@dataclass
class MockAttentionMetadata:
    """模拟vLLM的AttentionMetadata"""
    seq_lens: List[int]
    slot_mapping: torch.Tensor
    num_prefill_tokens: int


@dataclass
class MockModelInput:
    """模拟vLLM的ModelInput"""
    input_tokens: torch.Tensor
    attn_metadata: MockAttentionMetadata
    session_id: bytes = b"session_0000"
    layer_id: int = 0


def create_mock_model_input(seq_len: int = 10) -> MockModelInput:
    """创建模拟的模型输入"""
    # 创建输入tokens
    input_tokens = torch.randint(0, 1000, (seq_len,))
    
    # 创建注意力元数据
    # 槽位映射应该是一个二维张量，形状为(num_seqs, max_seq_len)
    # 对于单个序列，我们可以创建一个简单的映射
    slot_mapping = torch.arange(seq_len).unsqueeze(0)  # 形状: (1, seq_len)
    
    attn_metadata = MockAttentionMetadata(
        seq_lens=[seq_len],
        slot_mapping=slot_mapping,
        num_prefill_tokens=seq_len
    )
    
    return MockModelInput(
        input_tokens=input_tokens,
        attn_metadata=attn_metadata
    )


def create_mock_kv_caches(num_layers: int = 6, seq_len: int = 10, hidden_size: int = 64) -> List[torch.Tensor]:
    """创建模拟的KV缓存"""
    kv_caches = []
    for _ in range(num_layers):
        # 创建key和value缓存
        # 在vLLM中，KV缓存的形状通常是(2, num_blocks, block_size, num_heads, head_size)
        # 为了简化，我们创建一个形状为(2, seq_len, hidden_size)的张量
        key_cache = torch.randn(seq_len, hidden_size)
        value_cache = torch.randn(seq_len, hidden_size)
        # 组合成vLLM期望的格式 (2, seq_len, hidden_size)
        kv_cache = torch.stack([key_cache, value_cache], dim=0)
        kv_caches.append(kv_cache)
    return kv_caches


def test_kv_storage_and_retrieval():
    """测试KV存储和检索功能"""
    print("开始测试KV存储和检索功能...")
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    config = load_config_from_json(config_path)
    
    # 初始化引擎
    print("初始化KV引擎...")
    engine = init_engine(config=config)
    print(f"KV引擎初始化完成: {type(engine)}")
    
    # 创建模拟输入
    print("创建模拟模型输入...")
    model_input = create_mock_model_input(seq_len=10)
    print(f"模型输入创建完成，序列长度: {len(model_input.attn_metadata.seq_lens)}")
    
    # 创建模拟KV缓存
    print("创建模拟KV缓存...")
    kv_caches = create_mock_kv_caches(num_layers=6, seq_len=10)
    print(f"KV缓存创建完成，层数: {len(kv_caches)}")
    
    # 检查是否应该存储
    print("检查是否应该存储KV缓存...")
    store_status = should_store(model_input)
    print(f"存储状态: {store_status}")
    
    # 存储KV缓存
    if store_status == StoreStatus.STORED:
        print("开始存储KV缓存...")
        try:
            store_kv(
                model_config=None,
                parallel_config=None,
                transfer_config=None,
                model_executable=None,
                model_input=model_input,
                kv_caches=kv_caches,
                store_status=store_status
            )
            print("KV缓存存储完成")
        except Exception as e:
            print(f"存储KV缓存时出错: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 等待一段时间确保异步存储完成
    import time
    time.sleep(2)
    
    # 检查是否应该检索
    print("检查是否应该检索KV缓存...")
    retrieve_status = should_retrieve(model_input)
    print(f"检索状态: {retrieve_status}")
    
    # 检索KV缓存
    if retrieve_status == RetrieveStatus.HIT:
        print("开始检索KV缓存...")
        try:
            hidden_state, bypass_model_exec, _ = retrieve_kv(
                model_executable=None,
                model_input=model_input,
                kv_caches=kv_caches,
                retrieve_status=retrieve_status
            )
            print(f"KV缓存检索完成")
            print(f"隐藏状态: {hidden_state}")
            print(f"是否跳过模型执行: {bypass_model_exec}")
        except Exception as e:
            print(f"检索KV缓存时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("KV缓存未命中，无法检索")
    
    print("测试完成")


if __name__ == "__main__":
    test_kv_storage_and_retrieval()