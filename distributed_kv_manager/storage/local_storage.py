# distributed_kv_manager/storage/local_storage.py
import os
import io
import torch
import logging
from typing import Optional, Tuple
from .base import AbstractStorage

logger = logging.getLogger("LocalStorage")

class LocalStorage(AbstractStorage):
    """本地文件系统存储实现"""
    
    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info(f"Initialized LocalStorage with directory: {self.local_dir}")
        
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到本地文件系统"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(data)
                
            logger.debug(f"Successfully uploaded data to {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload data to {file_path}: {e}")
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        """从本地文件系统下载数据"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"File not found: {full_path}")
                return None
                
            with open(full_path, 'rb') as f:
                data = f.read()
                
            logger.debug(f"Successfully downloaded data from {full_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to download data from {file_path}: {e}")
            return None

    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        full_path = os.path.join(self.local_dir, file_path)
        return os.path.exists(full_path)

    def pack_kv_data(self, k_cache: torch.Tensor, v_cache: torch.Tensor, 
                    hidden: Optional[torch.Tensor], input_tokens: torch.Tensor, 
                    roi: torch.Tensor) -> bytes:
        """打包KV数据为字节流"""
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

    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据"""
        try:
            buffer = io.BytesIO(data)
            loaded = torch.load(buffer, map_location="cpu")
            return loaded["k_cache"], loaded["v_cache"], loaded.get("hidden", None)
        except Exception as e:
            logger.error(f"Failed to unpack KV data: {e}")
            return None, None, None
    
    def delete(self, file_path: str) -> bool:
        """删除文件（可选方法）"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.debug(f"Successfully deleted file: {full_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> list:
        """列出所有文件（可选方法）"""
        try:
            full_dir = os.path.join(self.local_dir, prefix)
            if not os.path.exists(full_dir):
                return []
                
            files = []
            for root, _, filenames in os.walk(full_dir):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), self.local_dir)
                    files.append(rel_path)
            return files
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []