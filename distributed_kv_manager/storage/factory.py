from typing import Optional
from .base import AbstractStorage
from .crail_storage import CrailStorage
from .local_storage import LocalStorage
from .caching_storage import CachingStorage

class StorageFactory:
    """存储后端工厂，根据配置创建不同的存储实例"""
    
    @staticmethod
    def create_storage(config) -> Optional[AbstractStorage]:
        """根据配置创建存储实例"""
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        enable_caching = getattr(config.kv_transfer_config, "enable_ssd_caching", True)
        cache_dir = getattr(config.kv_transfer_config, "ssd_cache_dir", "/tmp/ssd_cache")
        enable_prefetch = getattr(config.kv_transfer_config, "enable_prefetch", True)
        
        # 创建基础存储实例
        base_storage = None
        if storage_type == "crail":
            crail_dir = getattr(config.kv_transfer_config, "crail_dir", "/crail/kvcache")
            base_storage = CrailStorage(crail_dir)
        elif storage_type == "local":
            # 本地文件系统存储
            local_dir = getattr(config.kv_transfer_config, "local_dir", "/tmp/kvcache")
            base_storage = LocalStorage(local_dir)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        # 如果启用了缓存，则包装基础存储实例
        if enable_caching:
            return CachingStorage(base_storage, cache_dir, enable_prefetch)
        
        return base_storage