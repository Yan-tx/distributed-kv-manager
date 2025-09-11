import abc
from typing import Optional, Tuple
import torch

class AbstractStorage(abc.ABC):
    """抽象存储接口，所有存储后端都需要实现这些方法"""
    
    @abc.abstractmethod
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到存储"""
        pass
    
    @abc.abstractmethod
    def download(self, file_path: str) -> Optional[bytes]:
        """从存储下载数据"""
        pass
    
    @abc.abstractmethod
    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        pass
    
    @abc.abstractmethod
    def pack_kv_data(self, k_cache: torch.Tensor, v_cache: torch.Tensor, 
                    hidden: Optional[torch.Tensor], input_tokens: torch.Tensor, 
                    roi: torch.Tensor) -> bytes:
        """打包KV数据为字节流"""
        pass
    
    @abc.abstractmethod
    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据"""
        pass