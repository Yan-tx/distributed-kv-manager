#!/usr/bin/env python3
"""
Rust KV序列化模块测试
"""
import sys
import os
import time

# 获取项目根目录
project_root = os.path.join(os.path.dirname(__file__), '..')
project_root = os.path.abspath(project_root)

# 将Rust模块路径添加到Python路径
rust_module_path = os.path.join(project_root, 'rust_extensions', 'kv_serializer', 'target', 'release')
sys.path.append(rust_module_path)

print(f"项目根目录: {project_root}")
print(f"Rust模块路径: {rust_module_path}")

# 检查Rust模块文件是否存在
rust_module_file = os.path.join(rust_module_path, 'kv_serializer.dll')
print(f"Rust模块文件是否存在: {os.path.exists(rust_module_file)}")

def test_rust_module():
    """测试Rust模块是否正常工作"""
    try:
        import kv_serializer
        print("成功导入Rust模块")
        
        # 创建模拟的PyDict对象（简化实现）
        class MockPyDict:
            pass
        
        mock_dict = MockPyDict()
        
        # 测试序列化
        print("测试序列化...")
        packed = kv_serializer.pack_kv_data(mock_dict, mock_dict, mock_dict, mock_dict, mock_dict)
        print(f"序列化成功，数据大小: {len(packed)} 字节")
        
        # 测试反序列化
        print("测试反序列化...")
        unpacked = kv_serializer.unpack_kv_data(packed)
        print(f"反序列化成功，返回 {len(unpacked)} 个对象")
        
        return True
    except ImportError as e:
        print(f"无法导入Rust模块: {e}")
        # 尝试列出sys.path中的内容
        print("sys.path 内容:")
        for path in sys.path:
            print(f"  {path}")
        return False
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_rust_module():
    """基准测试Rust模块性能"""
    try:
        import kv_serializer
        
        # 创建模拟的PyDict对象（简化实现）
        class MockPyDict:
            pass
        
        mock_dict = MockPyDict()
        
        # 性能测试
        print("开始性能测试...")
        iterations = 1000
        start_time = time.time()
        
        for i in range(iterations):
            packed = kv_serializer.pack_kv_data(mock_dict, mock_dict, mock_dict, mock_dict, mock_dict)
            unpacked = kv_serializer.unpack_kv_data(packed)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"完成 {iterations} 次迭代")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"平均每次操作耗时: {avg_time*1000:.4f} 毫秒")
        
        return True
    except ImportError as e:
        print(f"无法导入Rust模块进行性能测试: {e}")
        return False
    except Exception as e:
        print(f"性能测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Rust KV序列化模块测试 ===")
    
    # 测试基本功能
    if test_rust_module():
        print("\n=== 基本功能测试通过 ===\n")
        
        # 进行性能测试
        print("=== 性能测试 ===")
        benchmark_rust_module()
        print("=== 性能测试完成 ===")
    else:
        print("\n=== 基本功能测试失败 ===")