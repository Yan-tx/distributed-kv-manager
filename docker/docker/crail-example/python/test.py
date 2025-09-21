# 创建文件: ~/sunshi/crail-example/python/test_lib.py
import os
import ctypes

# 打印当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 打印共享库路径
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcrail_jni.so")
print(f"共享库路径: {lib_path}")
print(f"共享库是否存在: {os.path.exists(lib_path)}")

try:
    # 尝试加载共享库
    lib = ctypes.CDLL(lib_path)
    print("成功加载共享库!")
    
    # 检查函数是否存在
    if hasattr(lib, "crail_upload_data"):
        print("找到 crail_upload_data 函数")
    else:
        print("未找到 crail_upload_data 函数")
        
except Exception as e:
    print(f"加载共享库失败: {e}")
