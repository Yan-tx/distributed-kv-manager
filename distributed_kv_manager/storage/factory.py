from typing import Optional
import logging
import importlib
from .base import AbstractStorage
from .crail_storage import CrailStorage
from .local_storage import LocalStorage
from .caching_storage import CachingStorage
from .composite_storage import CompositeStorage

logger = logging.getLogger("StorageFactory")


class _NoopRemoteStorage(AbstractStorage):
    """开发占位的远端存储实现：
    - upload: 返回 True（不持久化）
    - download: 返回 None
    - exists: 返回 False
    其余 pack/unpack 提供最小实现，避免被误用。
    仅用于在 composite 场景下“空出”远端实现。
    """

    def upload(self, file_path: str, data: bytes) -> bool:  # type: ignore[override]
        return True

    def download(self, file_path: str):  # type: ignore[override]
        return None

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        return False

    def pack_kv_data(self, *args, **kwargs) -> bytes:  # type: ignore[override]
        return b""

    def unpack_kv_data(self, data: bytes):  # type: ignore[override]
        return None, None


class StorageFactory:
    """存储后端工厂，根据配置创建不同的存储实例"""
    
    @staticmethod
    def create_storage(config) -> Optional[AbstractStorage]:
        """根据配置创建存储实例（支持内存/SSD两级缓存与预取模式）。"""
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        enable_caching = getattr(config.kv_transfer_config, "enable_ssd_caching", False)
        cache_dir = getattr(config.kv_transfer_config, "ssd_cache_dir", "/tmp/ssd_cache")
        enable_prefetch = getattr(config.kv_transfer_config, "enable_prefetch", True)
        cache_mode = getattr(config.kv_transfer_config, "cache_mode", None)  # none | only_mem | mem_and_ssd
        mem_capacity_bytes = getattr(config.kv_transfer_config, "mem_cache_capacity_bytes", 256 * 1024 * 1024)
        
        logger.info(f"创建存储实例: storage_type={storage_type}, enable_caching={enable_caching}, cache_dir={cache_dir}")
        
        # 创建基础存储实例
        base_storage = None
        if storage_type == "crail":
            crail_dir = getattr(config.kv_transfer_config, "crail_dir", "/crail/kvcache")
            base_storage = CrailStorage(crail_dir)
        elif storage_type == "local":
            # 本地文件系统存储
            local_dir = getattr(config.kv_transfer_config, "local_dir", "/tmp/kvcache")
            base_storage = LocalStorage(local_dir)
        elif storage_type == "composite":
            # 复合后端：本地 + 远端
            local_dir = getattr(config.kv_transfer_config, "local_dir", "/tmp/kvcache")
            # 远端路径优先使用 remote_dir，其次兼容旧字段 crail_dir
            remote_dir = getattr(
                config.kv_transfer_config,
                "remote_dir",
                getattr(config.kv_transfer_config, "crail_dir", "/crail/kvcache"),
            )
            front_n = getattr(config.kv_transfer_config, "layer_split_front", None)
            ratio = getattr(config.kv_transfer_config, "layer_split_ratio", 0.5)
            local_backend = LocalStorage(local_dir)

            # 远端后端抽象：优先使用 remote_backend_class（模块路径.类名）
            # 其次使用 remote_backend_type / remote_storage_type 指定的类型
            remote_backend_type = getattr(
                config.kv_transfer_config,
                "remote_backend_type",
                getattr(config.kv_transfer_config, "remote_storage_type", None),
            )
            remote_backend_class = getattr(
                config.kv_transfer_config, "remote_backend_class", None
            )

            def _import_backend(class_path: str):
                try:
                    module_name, cls_name = class_path.rsplit(".", 1)
                    mod = importlib.import_module(module_name)
                    return getattr(mod, cls_name)
                except Exception as e:
                    logger.warning("远端后端动态导入失败: %s (%s)", class_path, e)
                    return None

            if remote_backend_class:
                cls = _import_backend(str(remote_backend_class))
                remote_backend = cls(remote_dir) if cls else _NoopRemoteStorage()
            else:
                typ = None if remote_backend_type is None else str(remote_backend_type).lower()
                if typ in ("noop", "none", "placeholder"):
                    remote_backend = _NoopRemoteStorage()
                elif typ == "local":
                    remote_backend = LocalStorage(remote_dir)
                elif typ == "crail" or (
                    typ is None and hasattr(config.kv_transfer_config, "crail_dir")
                ):
                    # 兼容默认：未指定时沿用 Crail 作为远端实现
                    remote_backend = CrailStorage(remote_dir)
                elif typ and "." in typ:
                    cls = _import_backend(typ)
                    remote_backend = cls(remote_dir) if cls else _NoopRemoteStorage()
                else:
                    # 未指定或未知类型，使用占位远端
                    remote_backend = _NoopRemoteStorage()

            # 如果远端为占位且未指定前段层数，强制全部前段落本地，避免读取空后段
            if isinstance(remote_backend, _NoopRemoteStorage) and front_n is None:
                front_n = 1_000_000
            base_storage = CompositeStorage(local_backend, remote_backend,
                                            layer_split_front=front_n,
                                            layer_split_ratio=ratio)
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        
        # 兼容旧行为：无 cache_mode 字段，则遵循 enable_ssd_caching 逻辑
        if cache_mode is None:
            if enable_caching:
                logger.info(f"启用SSD缓存（兼容模式），目录: {cache_dir}")
                return CachingStorage(base_storage, cache_dir, enable_prefetch, cache_mode=None,
                                      mem_capacity_bytes=mem_capacity_bytes)
            logger.info("未启用缓存（兼容模式），直接返回基础存储实例")
            return base_storage

        # 新模式：按 cache_mode 精确控制
        mode = str(cache_mode).lower()
        if mode == "none":
            logger.info("缓存模式: none（仅基础存储）")
            return base_storage
        elif mode == "only_mem":
            logger.info(f"缓存模式: only_mem（内存缓存，容量={mem_capacity_bytes}B）")
            return CachingStorage(base_storage, cache_dir, enable_prefetch,
                                  cache_mode="only_mem", mem_capacity_bytes=mem_capacity_bytes)
        elif mode == "mem_and_ssd":
            logger.info(f"缓存模式: mem_and_ssd（内存+SSD，SSD目录={cache_dir}，容量={mem_capacity_bytes}B）")
            return CachingStorage(base_storage, cache_dir, enable_prefetch,
                                  cache_mode="mem_and_ssd", mem_capacity_bytes=mem_capacity_bytes)
        else:
            logger.warning(f"未知 cache_mode: {cache_mode}，回退到基础存储")
            return base_storage