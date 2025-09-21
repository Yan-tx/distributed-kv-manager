import os
import io
import time
import ctypes
import torch
from typing import List, Tuple, Dict, Any, Optional

# 加载JNI库
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcrail_jni.so")
_lib = ctypes.CDLL(_lib_path)

# 定义函数原型
_lib.crail_upload_data.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.crail_upload_data.restype = ctypes.c_bool

_lib.crail_download_data.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
_lib.crail_download_data.restype = ctypes.c_void_p

_lib.crail_list_directory.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
_lib.crail_list_directory.restype = ctypes.POINTER(ctypes.c_char_p)

_lib.free_string_array.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
_lib.free_string_array.restype = None

class CrailKVCache:
    """使用JNI接口管理Crail KV缓存"""
    
    @staticmethod
    def save_kv_cache(kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
                      crail_path: str, 
                      max_seq_len: Optional[int] = None, 
                      prompt_len: Optional[int] = None,
                      model_config: Optional[Dict[str, Any]] = None) -> bool:
        """保存KV缓存到Crail，使用JNI接口"""
        try:
            print(f"正在保存KV cache到Crail: {crail_path}")
            start_time = time.time()
            
            # 构建保存数据
            save_data = {
                "kv_caches": [(k.detach().cpu(), v.detach().cpu()) for k, v in kv_caches],
                "timestamp": time.time(),
                "max_seq_len": max_seq_len,
                "prompt_len": prompt_len,
                "model_config": model_config
            }

            # 序列化到内存缓冲区
            buffer = io.BytesIO()
            torch.save(save_data, buffer)
            data_bytes = buffer.getvalue()
            
            # 使用JNI上传到Crail
            data_ptr = ctypes.cast(data_bytes, ctypes.c_void_p)
            result = _lib.crail_upload_data(
                crail_path.encode('utf-8'), 
                data_ptr, 
                len(data_bytes)
            )
            
            duration = time.time() - start_time
            
            if result:
                print(f"KV cache已成功上传到Crail: {crail_path}, 用时: {duration:.2f}秒")
            else:
                print(f"上传KV cache到Crail失败: {crail_path}")
                
            return result
            
        except Exception as e:
            print(f"保存KV cache到Crail失败: {str(e)}")
            return False
    
    @staticmethod
    def load_kv_cache(crail_path: str, device: torch.device) -> Tuple[Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Dict[str, Any]]:
        """从Crail加载KV缓存，使用JNI接口"""
        try:
            print(f"正在从Crail加载KV cache: {crail_path}")
            start_time = time.time()
            
            # 使用JNI从Crail下载
            data_size = ctypes.c_size_t()
            data_ptr = _lib.crail_download_data(crail_path.encode('utf-8'), ctypes.byref(data_size))
            
            if not data_ptr or data_size.value == 0:
                print(f"无法从Crail下载数据: {crail_path}")
                return None, {}
                
            # 从内存中加载数据
            data_bytes = ctypes.string_at(data_ptr, data_size.value)
            buffer = io.BytesIO(data_bytes)
            
            # 加载PyTorch数据
            saved_data = torch.load(buffer)
            
            # 释放C分配的内存
            libc = ctypes.CDLL(None)
            libc.free(data_ptr)
            
            # 验证并获取元数据
            metadata = {
                "original_max_seq_len": saved_data.get("max_seq_len"),
                "original_prompt_len": saved_data.get("prompt_len"),
                "timestamp": saved_data.get("timestamp"),
                "model_config": saved_data.get("model_config", {})
            }
            
            # 将KV cache移动到指定设备
            kv_caches = []
            for k, v in saved_data["kv_caches"]:
                kv_caches.append((k.to(device), v.to(device)))
            
            duration = time.time() - start_time
            print(f"成功从Crail加载KV cache: {crail_path}, 用时: {duration:.2f}秒")
            return kv_caches, metadata
            
        except Exception as e:
            print(f"从Crail加载KV cache失败: {str(e)}")
            return None, {}
    
    @staticmethod
    def list_kv_caches(directory: str) -> List[str]:
        """列出Crail目录中的条目"""
        try:
            count = ctypes.c_int()
            entries_ptr = _lib.crail_list_directory(directory.encode('utf-8'), ctypes.byref(count))
            
            if not entries_ptr or count.value == 0:
                return []
                
            # 从C字符串数组转换为Python列表
            result = []
            for i in range(count.value):
                entry = ctypes.cast(entries_ptr[i], ctypes.c_char_p).value.decode('utf-8')
                result.append(entry)
            
            # 释放C分配的内存
            _lib.free_string_array(entries_ptr, count.value)
            
            return result
            
        except Exception as e:
            print(f"列出目录失败: {str(e)}")
            return []
