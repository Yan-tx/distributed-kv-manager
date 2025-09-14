import torch
from distributed_kv_manager.engine.kv_engine import KVEngine
from types import SimpleNamespace

# ------------------ 模拟配置 ------------------ #
config = SimpleNamespace()
config.kv_transfer_config = SimpleNamespace(crail_dir="./test_crail_kvcache")
config.rank = 0
config.local_rank = 0

# ------------------ 初始化 Engine ------------------ #
engine = CrailKVEngine(config)

# ------------------ 构造测试数据 ------------------ #
seq_len = 5
num_layers = 2
num_heads = 2
head_size = 4

input_tokens = torch.arange(seq_len)
kv_caches = [torch.randn(2, seq_len, num_heads, head_size) for _ in range(num_layers)]
hidden_states = torch.randn(seq_len, 8)

# 模拟 model_input
class ModelInput:
    def __init__(self, input_tokens, hidden_states):
        self.input_tokens = input_tokens
        self.hidden_states = hidden_states
        self.attn_metadata = SimpleNamespace(
            seq_lens=[seq_len],
            slot_mapping=torch.arange(seq_len).unsqueeze(0),
            num_prefill_tokens=seq_len
        )

model_input = ModelInput(input_tokens, hidden_states)

# ------------------ 测试存储 KV ------------------ #
print("存储 KV 到 Crail ...")
engine.store_kv(None, None, None, None, model_input, kv_caches, None)

# 等待异步任务完成
engine.close()
print("存储完成。")

# ------------------ 测试检索 KV ------------------ #
engine = CrailKVEngine(config)  # 新建实例
retrieve_status = engine.should_retrieve(model_input)
retrieved_hidden, bypass, _ = engine.retrieve_kv(None, model_input, kv_caches, retrieve_status)

if retrieved_hidden is not None:
    print("成功检索 KV 缓存，跳过模型计算。")
    print("retrieved_hidden:", retrieved_hidden)
else:
    print("未能检索 KV 缓存，需要执行模型计算。")

# ------------------ 检查存在 ------------------ #
file_path = engine._make_key(input_tokens)
exists = engine._crail_exists(file_path)
print(f"检查缓存存在: {exists}")

# ------------------ 清理 ------------------ #
engine.close()
print("Engine 已关闭。")
