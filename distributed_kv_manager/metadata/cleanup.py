import time
import threading
import logging
from typing import List, Optional
from .etcd import KVMetadataManager, KVMetadata

logger = logging.getLogger("KVCleanup")


class KVCleanupManager:
    """KV缓存清理管理器，负责定期清理过期的KV缓存"""
    
    def __init__(self, meta_manager: KVMetadataManager, cleanup_interval: int = 3600, storage=None):
        """
        初始化清理管理器
        
        Args:
            meta_manager: 元数据管理器
            cleanup_interval: 清理间隔时间（秒），默认1小时
            storage: 存储后端实例，用于删除KV数据文件
        """
        self.meta_manager = meta_manager
        self.cleanup_interval = cleanup_interval
        self.storage = storage
        self._stop_flag = False
        self._thread = None
        self._lock = threading.Lock()
        
    def start(self):
        """启动后台清理线程"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("清理线程已经在运行中")
            return
            
        self._stop_flag = False
        self._thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._thread.start()
        logger.info(f"KV缓存清理线程已启动，清理间隔: {self.cleanup_interval}秒")
        
    def stop(self):
        """停止后台清理线程"""
        with self._lock:
            self._stop_flag = True
            
        if self._thread and self._thread.is_alive():
            self._thread.join()
            logger.info("KV缓存清理线程已停止")
            
    def _cleanup_worker(self):
        """清理工作线程"""
        while not self._stop_flag:
            try:
                # 执行清理
                self._perform_cleanup()
                
                # 等待下次清理
                for _ in range(self.cleanup_interval):
                    if self._stop_flag:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"清理过程中发生错误: {e}")
                
    def _perform_cleanup(self):
        """执行一次清理操作"""
        logger.info("开始执行KV缓存清理")
        try:
            # 获取所有元数据键
            all_keys = self._scan_all_metadata_keys()
            logger.info(f"扫描到 {len(all_keys)} 个元数据键")
            
            cleaned_count = 0
            current_time = int(time.time())
            
            for key in all_keys:
                try:
                    logger.debug(f"检查元数据键: {key}")
                    # 直接使用原始键获取元数据（扫描返回的已经是完整键）
                    meta = self.meta_manager.get_metadata_by_full_key(key)
                    if meta:
                        logger.debug(f"元数据详情 - 创建时间: {meta.create_time}, 最后访问: {meta.last_access}, 过期时间: {meta.expire_time}")
                        # 检查是否过期
                        if meta.is_expired():
                            logger.info(f"发现过期KV缓存: {key}")
                            # 清理过期的KV缓存
                            self._cleanup_expired_kv(meta)
                            cleaned_count += 1
                        else:
                            logger.debug(f"KV缓存未过期: {key}")
                    else:
                        logger.warning(f"未找到元数据: {key}，可能已被删除或键格式不正确")
                except Exception as e:
                    logger.error(f"处理元数据 {key} 时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                    
            logger.info(f"KV缓存清理完成，共清理 {cleaned_count} 个过期项")
        except Exception as e:
            logger.error(f"执行清理操作时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
    def _scan_all_metadata_keys(self) -> List[str]:
        """
        扫描所有元数据键
        """
        try:
            return self.meta_manager.scan_all_metadata_keys()
        except Exception as e:
            logger.error(f"扫描元数据键时发生错误: {e}")
            return []
        
    def _cleanup_expired_kv(self, meta: KVMetadata):
        """
        清理过期的KV缓存
        
        Args:
            meta: 过期的元数据
        """
        try:
            # 1. 删除存储中的KV数据文件
            if self.storage:
                try:
                    success = self.storage.delete(meta.file_path)
                    if success:
                        logger.debug(f"已从存储中删除过期KV数据文件: {meta.file_path}")
                    else:
                        logger.warning(f"未能从存储中删除过期KV数据文件: {meta.file_path}")
                except Exception as e:
                    logger.error(f"删除存储中的KV数据文件 {meta.file_path} 时发生错误: {e}")
            
            # 2. 从ETCD删除元数据
            self.meta_manager.delete_metadata(meta.file_path)
            
            # 3. 记录日志
            logger.info(f"已清理过期KV缓存: {meta.file_path}")
        except Exception as e:
            logger.error(f"清理过期KV缓存 {meta.file_path} 时发生错误: {e}")