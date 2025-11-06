import os
import io
import torch
import logging
import threading
import hashlib
import re
from typing import Optional, Tuple, List
from .base import AbstractStorage

logger = logging.getLogger("CachingStorage")


class SSDCache:
    """SSD缓存实现，用于存储KV数据"""
    
    def __init__(self, cache_dir: str = "/tmp/ssd_cache"):
        self.cache_dir = cache_dir
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[bytes]:
        """获取缓存数据"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = f.read()
                logger.debug(f"Hit SSD cache for {key}")
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to load data from SSD cache {key}: {e}")
            return None
            
    def put(self, key: str, value: bytes) -> bool:
        """放入缓存数据"""
        try:
            cache_path = self._get_cache_path(key)
            # 确保目录存在
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(value)
            logger.debug(f"Saved data to SSD cache: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save data to SSD cache {key}: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """删除缓存数据"""
        try:
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.debug(f"Deleted data from SSD cache: {cache_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete data from SSD cache {key}: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """检查缓存中是否存在"""
        cache_path = self._get_cache_path(key)
        return os.path.exists(cache_path)
        
    def _get_cache_path(self, file_path: str) -> str:
        """获取文件在缓存中的路径"""
        # 保持原有的文件路径结构，但基于缓存目录
        # 移除开头的斜杠以避免路径问题
        if file_path.startswith("/"):
            file_path = file_path[1:]
        return os.path.join(self.cache_dir, file_path)


class CachingStorage(AbstractStorage):
    """带SSD缓存功能的存储包装器"""
    
    def __init__(self, storage_backend: AbstractStorage, cache_dir: str = "/tmp/ssd_cache", 
                 enable_prefetch: bool = True):
        self.storage_backend = storage_backend
        self.ssd_cache = SSDCache(cache_dir)
        self.enable_prefetch = enable_prefetch
        
        # 预取相关
        self.prefetch_thread = None
        self.prefetch_queue = []
        self.prefetch_lock = threading.Lock()
        
        logger.info(f"Initialized CachingStorage with cache_dir: {cache_dir}")
    
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据，同时更新缓存"""
        # 先上传到后端存储
        success = self.storage_backend.upload(file_path, data)
        if success:
            # 更新SSD缓存
            self.ssd_cache.put(file_path, data)
            logger.debug(f"数据已上传并缓存: {file_path}")
        else:
            logger.error(f"数据上传失败: {file_path}")
        return success
    
    def download(self, file_path: str) -> Optional[bytes]:
        """下载数据，优先从SSD缓存获取"""
        # 1. 先检查SSD缓存
        data = self.ssd_cache.get(file_path)
        if data is not None:
            logger.debug(f"缓存命中: {file_path}")
            # 启动预取
            if self.enable_prefetch:
                self._trigger_prefetch(file_path)
            return data
            
        # 2. 从后端存储获取
        logger.debug(f"缓存未命中，从后端存储获取: {file_path}")
        data = self.storage_backend.download(file_path)
        if data is not None:
            # 保存到SSD缓存
            self.ssd_cache.put(file_path, data)
            logger.debug(f"数据已从后端存储获取并缓存: {file_path}")
            # 启动预取
            if self.enable_prefetch:
                self._trigger_prefetch(file_path)
        else:
            logger.warning(f"无法从后端存储获取数据: {file_path}")
        return data
    
    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        # 检查缓存中是否存在
        if self.ssd_cache.exists(file_path):
            return True
            
        # 检查后端存储
        return self.storage_backend.exists(file_path)
    
    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        """打包KV数据为字节流"""
        return self.storage_backend.pack_kv_data(k_cache, v_cache, input_tokens, roi)
    
    def unpack_kv_data(
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据"""
        return self.storage_backend.unpack_kv_data(data)
    
    def _trigger_prefetch(self, accessed_file: str) -> None:
        """触发预取机制"""
        # 将预取任务加入队列，在后台线程处理
        with self.prefetch_lock:
            self.prefetch_queue.append(accessed_file)
            if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
                self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
                self.prefetch_thread.start()
    
    def _prefetch_worker(self) -> None:
        """预取工作线程"""
        while True:
            with self.prefetch_lock:
                if not self.prefetch_queue:
                    break
                file_path = self.prefetch_queue.pop(0)
            
            # 预取相关文件
            self._prefetch_related_files(file_path)
    
    def _prefetch_related_files(self, file_path: str) -> None:
        """预取相关文件"""
        # 根据KV引擎的文件命名规则实现预取策略
        # 新的文件命名规则: /{storage_dir}/kv_{session_str}_layer_{layer_id}_{seq_hash}.pt
        
        # 生成预取文件列表
        prefetch_files = []
        
        # 1. 预取同一会话后续层的文件
        session_layer_files = self._generate_session_layer_files(file_path)
        prefetch_files.extend(session_layer_files)
        
        # 2. 预取同一目录下最近访问的其他KV文件
        recent_files = self._generate_recent_files(os.path.dirname(file_path), exclude=file_path)
        prefetch_files.extend(recent_files)
        
        # 预取这些文件
        for prefetch_file in prefetch_files:
            if not self.ssd_cache.exists(prefetch_file):
                # 异步预取
                threading.Thread(target=self._prefetch_single_file, args=(prefetch_file,), daemon=True).start()
    
    def _generate_session_layer_files(self, file_path: str) -> List[str]:
        """生成同一会话后续层的文件名"""
        # 从文件路径中提取会话和层信息
        # 文件命名规则: kv_{session_str}_layer_{layer_id}_{seq_hash}.pt
        basename = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        
        # 解析文件名获取session和layer信息
        match = re.match(r"kv_(.+)_layer_(\d+)_(.+)\.pt", basename)
        if not match:
            return []
            
        session_str, layer_id, seq_hash = match.groups()
        current_layer = int(layer_id)
        
        # 生成同一会话后续层的文件名
        session_files = []
        # 预取接下来几层的文件（假设最多28层）
        for next_layer in range(current_layer + 1, min(current_layer + 6, 28)):
            next_file = f"kv_{session_str}_layer_{next_layer}_{seq_hash}.pt"
            session_files.append(os.path.join(dir_name, next_file))
            
        return session_files
    
    def _generate_recent_files(self, dir_path: str, exclude: Optional[str] = None) -> List[str]:
        """生成目录下最近修改的文件列表"""
        try:
            # 获取目录下所有KV文件
            kv_files = []
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    if filename.startswith("kv_") and filename.endswith(".pt"):
                        file_path = os.path.join(dir_path, filename)
                        # 跳过排除的文件
                        if exclude and file_path == exclude:
                            continue
                        mtime = os.path.getmtime(file_path)
                        kv_files.append((file_path, mtime))
            
            # 按修改时间排序，获取最近的文件
            kv_files.sort(key=lambda x: x[1], reverse=True)
            
            # 返回最近的几个文件路径
            return [file_path for file_path, _ in kv_files[:3]]
        except Exception as e:
            logger.error(f"Error generating recent files in {dir_path}: {e}")
            return []
    
    def _prefetch_single_file(self, file_path: str) -> None:
        """预取单个文件"""
        try:
            # 检查是否已经在缓存中
            if self.ssd_cache.exists(file_path):
                return
                
            # 从后端存储获取
            data = self.storage_backend.download(file_path)
            if data is not None:
                # 保存到SSD缓存
                self.ssd_cache.put(file_path, data)
                logger.debug(f"Prefetched file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to prefetch file {file_path}: {e}")