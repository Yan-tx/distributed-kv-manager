import sys
import os
import time
import torch
import io

# 将Rust模块路径添加到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'rust_extensions', 'kv_serializer', 'target', 'release'))

try:
    import kv_serializer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("警告: 无法导入Rust模块，将使用Python原生实现")

def python_pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi):
    """Python原生实现的KV数据打包"""
    data = {
        "k_cache": k_cache.cpu(),
        "v_cache": v_cache.cpu(),
        "hidden": hidden.cpu() if hidden is not None else None,
        "input_tokens": input_tokens.cpu(),
        "roi": roi.cpu()
    }
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()

def python_unpack_kv_data(data):
    """Python原生实现的KV数据解包"""
    try:
        buffer = io.BytesIO(data)
        loaded = torch.load(buffer, map_location="cpu")
        return loaded["k_cache"], loaded["v_cache"], loaded.get("hidden", None)
    except Exception as e:
        print(f"Failed to unpack KV data: {e}")
        return None, None, None

def create_test_data():
    """创建测试数据"""
    # 创建测试张量
    k_cache = torch.randn(100, 64, 128)
    v_cache = torch.randn(100, 64, 128)
    hidden = torch.randn(100, 512)
    input_tokens = torch.randint(0, 1000, (100,))
    roi = torch.ones(100, dtype=torch.bool)
    
    return k_cache, v_cache, hidden, input_tokens, roi

def benchmark_serialization():
    """性能测试"""
    if not RUST_AVAILABLE:
        print("Rust模块不可用，跳过性能测试")
        return
    
    print("创建测试数据...")
    k_cache, v_cache, hidden, input_tokens, roi = create_test_data()
    
    # 创建模拟的PyDict对象（简化实现）
    class MockPyDict:
        pass
    
    mock_dict = MockPyDict()
    
    # 测试Python实现
    print("测试Python实现...")
    start_time = time.time()
    for _ in range(100):
        packed = python_pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi)
        unpacked = python_unpack_kv_data(packed)
    python_time = time.time() - start_time
    print(f"Python实现耗时: {python_time:.4f}秒")
    
    # 测试Rust实现
    print("测试Rust实现...")
    start_time = time.time()
    for _ in range(100):
        packed = kv_serializer.pack_kv_data(mock_dict, mock_dict, mock_dict, mock_dict, mock_dict)
        unpacked = kv_serializer.unpack_kv_data(packed)
    rust_time = time.time() - start_time
    print(f"Rust实现耗时: {rust_time:.4f}秒")
    
    # 计算性能提升
    if rust_time > 0:
        speedup = python_time / rust_time
        print(f"性能提升: {speedup:.2f}倍")
    else:
        print("Rust实现耗时为0，无法计算性能提升")

if __name__ == "__main__":
    benchmark_serialization()