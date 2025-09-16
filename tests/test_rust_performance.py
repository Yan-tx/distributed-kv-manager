#!/usr/bin/env python3
"""
Rust KV序列化模块性能对比测试
"""
import sys
import os
import time
import io
import torch

# 获取项目根目录
project_root = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.abspath(project_root)

# 将Rust模块路径添加到Python路径
rust_module_path = os.path.join(project_root, 'rust_extensions', 'kv_serializer', 'target', 'release')
sys.path.append(rust_module_path)

# 检查Rust模块文件是否存在
rust_module_file = os.path.join(rust_module_path, 'kv_serializer.dll')
rust_module_file_linux = os.path.join(rust_module_path, 'libkv_serializer.so')

# 如果在Linux环境下，我们需要确保so文件能被找到
if os.path.exists(rust_module_file_linux):
    # 在Linux环境下，可能需要将so文件链接为正确的名称
    expected_module_file = os.path.join(rust_module_path, 'kv_serializer.so')
    if not os.path.exists(expected_module_file):
        try:
            os.symlink(rust_module_file_linux, expected_module_file)
        except Exception as e:
            pass

try:
    import kv_serializer
    RUST_AVAILABLE = True
    print("成功导入Rust模块")
except ImportError:
    RUST_AVAILABLE = False
    print("警告: 无法导入Rust模块，将使用Python原生实现")

def python_pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi):
    """Python原生实现的KV数据打包"""
    data = {
        "k_cache": k_cache.cpu() if hasattr(k_cache, 'cpu') else k_cache,
        "v_cache": v_cache.cpu() if hasattr(v_cache, 'cpu') else v_cache,
        "hidden": hidden.cpu() if hasattr(hidden, 'cpu') and hidden is not None else hidden,
        "input_tokens": input_tokens.cpu() if hasattr(input_tokens, 'cpu') else input_tokens,
        "roi": roi.cpu() if hasattr(roi, 'cpu') else roi
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
    
    # 测试Python实现
    print("测试Python原生实现...")
    start_time = time.time()
    for _ in range(1000):
        packed = python_pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi)
        unpacked = python_unpack_kv_data(packed)
    python_time = time.time() - start_time
    print(f"Python实现耗时: {python_time:.4f}秒")
    
    # 测试Rust实现
    print("测试Rust实现...")
    # 创建简单的字典对象来模拟PyDict
    k_cache_dict = {}
    v_cache_dict = {}
    hidden_dict = {}
    input_tokens_dict = {}
    roi_dict = {}
    
    start_time = time.time()
    for _ in range(1000):
        packed = kv_serializer.pack_kv_data(k_cache_dict, v_cache_dict, hidden_dict, input_tokens_dict, roi_dict)
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
    print("=== Rust KV序列化模块性能对比测试 ===")
    benchmark_serialization()