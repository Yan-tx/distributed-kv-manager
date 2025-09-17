import torch
from distributed_kv_manager import init_engine, destroy_engine, load_config_from_json
from distributed_kv_manager.engine import StoreStatus, RetrieveStatus
from distributed_kv_manager.storage.factory import StorageFactory

def test_config_loading():
    \"\"\"测试从config.json加载配置\"\"\"
    print(\"=== 测试从config.json加载配置 ===\")
    
    # 从config.json加载配置
    config = load_config_from_json()
    print(f\"加载的配置: {config}\")
    print(f\"存储类型: {getattr(config.kv_transfer_config, 'storage_type', 'N/A')}\")
    print(f\"存储目录: {getattr(config.kv_transfer_config, 'storage_dir', 'N/A')}\")
    print(f\"ETCD端点: {getattr(config.kv_transfer_config, 'etcd_endpoints', 'N/A')}\")
    print(f\"Crail目录: {getattr(config.kv_transfer_config, 'crail_dir', 'N/A')}\")
    print(f\"本地目录: {getattr(config.kv_transfer_config, 'local_dir', 'N/A')}\")
    
    # 测试存储工厂
    print(\"\\n=== 测试存储工厂 ===\")
    try:
        storage = StorageFactory.create_storage(config)
        print(f\"存储实例创建成功: {type(storage).__name__}\")
        
        # 测试存储功能
        test_data = b\"test data for storage\"
        test_file = \"test_config_file.txt\"
        
        # 上传测试
        upload_success = storage.upload(test_file, test_data)
        print(f\"上传测试结果: {'成功' if upload_success else '失败'}\")
        
        # 存在性测试
        exists = storage.exists(test_file)
        print(f\"文件存在性检查: {'存在' if exists else '不存在'}\")
        
        # 下载测试
        if exists:
            downloaded_data = storage.download(test_file)
            print(f\"下载测试结果: {'成功' if downloaded_data else '失败'}\")
            if downloaded_data:
                print(f\"数据一致性检查: {'一致' if downloaded_data == test_data else '不一致'}\")
        
        # 删除测试文件
        if hasattr(storage, 'delete'):
            storage.delete(test_file)
            print(\"测试文件已清理\")
            
    except Exception as e:
        print(f\"存储实例创建或测试失败: {e}\")
    
    # 初始化引擎
    print(\"\\n=== 测试引擎初始化 ===\")
    try:
        engine = init_engine(config)
        print(f\"引擎初始化成功: {type(engine).__name__}\")
        
        # 检查存储实例
        print(f\"引擎存储实例类型: {type(engine._storage).__name__}\")
        
        # 构造测试数据
        seq_len = 3
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
                self.attn_metadata = type('AttnMetadata', (), {
                    'seq_lens': [seq_len],
                    'slot_mapping': torch.arange(seq_len).unsqueeze(0),
                    'num_prefill_tokens': seq_len
                })()

        model_input = ModelInput(input_tokens, hidden_states)
        
        # 测试should_store方法
        store_status = engine.should_store(model_input)
        print(f\"Should store状态: {store_status}\")
        
        # 清理引擎
        destroy_engine()
        print(\"引擎已销毁\")
        
    except Exception as e:
        print(f\"引擎初始化或测试失败: {e}\")
    
    print(\"\\n=== 测试完成 ===\")

if __name__ == \"__main__\":
    test_config_loading()