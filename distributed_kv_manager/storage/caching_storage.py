import os
import io
import torch
import logging
import threading
import hashlib
import re
import time
from typing import Optional, Tuple, List
from .base import AbstractStorage
from ..prefetch import (
    PrefetchBuffer,
    BudgetEstimator,
    RateLimiter,
    IOAggregator,
    PlanBuilder,
)

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


class MemoryCache:
    """简单的基于内存的LRU缓存，按字节容量驱动淘汰。

    存储值为 bytes（与后端/SSD一致），线程安全。
    """

    def __init__(self, capacity_bytes: int = 256 * 1024 * 1024):
        from collections import OrderedDict
        self.capacity = max(int(capacity_bytes), 1)
        self._map = OrderedDict()  # key -> (bytes, size)
        self._size = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            v = self._map.get(key)
            if v is None:
                return None
            data, size = v
            # LRU 触顶
            self._map.move_to_end(key)
            return data

    def put(self, key: str, data: bytes) -> bool:
        if data is None:
            return False
        size = len(data)
        with self._lock:
            # 更新或新增
            old = self._map.pop(key, None)
            if old is not None:
                self._size -= old[1]
            self._map[key] = (data, size)
            self._map.move_to_end(key)
            self._size += size
            # 淘汰
            evicted = 0
            while self._size > self.capacity and self._map:
                k, (d, s) = self._map.popitem(last=False)
                self._size -= s
                evicted += 1
            if evicted:
                logger.debug(f"MemoryCache evicted {evicted} entries, size={self._size}")
            return True

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._map

    def delete(self, key: str) -> bool:
        with self._lock:
            v = self._map.pop(key, None)
            if v is None:
                return False
            self._size -= v[1]
            return True


class CachingStorage(AbstractStorage):
    """分层缓存存储包装器：支持内存与SSD两级缓存，以及按窗口聚合的预取策略。"""

    def __init__(self, storage_backend: AbstractStorage, cache_dir: str = "/tmp/ssd_cache",
                 enable_prefetch: bool = True, cache_mode: Optional[str] = None,
                 mem_capacity_bytes: int = 256 * 1024 * 1024):
        self.storage_backend = storage_backend
        self.ssd_cache = SSDCache(cache_dir)
        self.mem_cache = MemoryCache(mem_capacity_bytes)
        self.enable_prefetch = enable_prefetch

        # 缓存模式：none / only_mem / mem_and_ssd
        mode = (cache_mode or "auto").lower()
        # auto 表示保持兼容：如果调用方仍使用旧接口（无cache_mode但开启了SSD），则启用SSD但内存缓存关闭
        if mode == "none":
            self._use_mem = False
            self._use_ssd = False
        elif mode == "only_mem":
            self._use_mem = True
            self._use_ssd = False
        elif mode == "mem_and_ssd":
            self._use_mem = True
            self._use_ssd = True
        else:
            # 兼容旧行为：依赖调用方是否“打算使用SSD包装器”，此类实例即视为 _use_ssd=True, _use_mem=False
            self._use_mem = False
            self._use_ssd = True
        
        # 预取与聚合（策略1+2最小实现）
        self._buffer = PrefetchBuffer(capacity=4096)
        self._estimator = BudgetEstimator()
        # 默认为 60% 带宽份额
        self._rate_limiter = RateLimiter(bytes_per_sec=self._estimator.bandwidth() * 0.6)
        self._planner = PlanBuilder()
        # 窗口聚合 + QD 限制
        self._aggregator = IOAggregator(
            fetch_fn=self._fetch_to_cache,
            on_ready=self._on_prefetch_ready,
            rate_limiter=self._rate_limiter,
            window_ms=30,
            max_batch_bytes=64 * 1024 * 1024,
            max_qd=4,
        ) if self.enable_prefetch else None
        
        logger.info(f"Initialized CachingStorage with cache_dir: {cache_dir}")
    
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据，同时更新缓存"""
        # 先上传到后端存储
        success = self.storage_backend.upload(file_path, data)
        if success:
            # 更新缓存
            if self._use_ssd:
                self.ssd_cache.put(file_path, data)
            if self._use_mem:
                self.mem_cache.put(file_path, data)
            logger.debug(f"数据已上传并缓存: {file_path}")
        else:
            logger.error(f"数据上传失败: {file_path}")
        return success
    
    def download(self, file_path: str) -> Optional[bytes]:
        """下载数据，优先从SSD缓存获取"""
        # 1) 先查内存缓存
        data = self.mem_cache.get(file_path) if self._use_mem else None
        if data is not None:
            logger.debug(f"内存缓存命中: {file_path}")
            try:
                self._buffer.on_access(file_path, hit=True)
            except Exception:
                pass
            if self.enable_prefetch and self._aggregator is not None:
                self._trigger_prefetch(file_path, front_load_time_sec=0.0, bytes_observed=len(data))
            return data

        # 2) 再查SSD缓存
        data = self.ssd_cache.get(file_path) if self._use_ssd else None
        if data is not None:
            logger.debug(f"SSD缓存命中: {file_path}")
            # 回填内存
            if self._use_mem:
                self.mem_cache.put(file_path, data)
            # 访问反馈
            try:
                self._buffer.on_access(file_path, hit=True)
            except Exception:
                pass
            # 启动策略1/2：基于当前命中预测后续项并聚合
            if self.enable_prefetch and self._aggregator is not None:
                self._trigger_prefetch(file_path, front_load_time_sec=0.0, bytes_observed=len(data))
            return data
            
        # 3) 从后端存储获取
        logger.debug(f"缓存未命中，从后端存储获取: {file_path}")
        t0 = time.time()
        data = self.storage_backend.download(file_path)
        dt = max(time.time() - t0, 0.0)
        if data is not None:
            # 保存到SSD缓存
            if self._use_ssd:
                self.ssd_cache.put(file_path, data)
            if self._use_mem:
                self.mem_cache.put(file_path, data)
            logger.debug(f"数据已从后端存储获取并缓存: {file_path}")
            # 更新带宽观测
            try:
                self._estimator.update(len(data), dt)
                # 动态调整限速（60% 份额）
                self._rate_limiter.set_rate(self._estimator.bandwidth() * 0.6)
            except Exception:
                pass
            # 启动预取
            if self.enable_prefetch and self._aggregator is not None:
                self._trigger_prefetch(file_path, front_load_time_sec=dt, bytes_observed=len(data))
        else:
            logger.warning(f"无法从后端存储获取数据: {file_path}")
            try:
                self._buffer.on_access(file_path, hit=False)
            except Exception:
                pass
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
    
    def _trigger_prefetch(self, accessed_file: str, front_load_time_sec: float, bytes_observed: int) -> None:
        """构建计划（策略1）并通过聚合器提交（策略2）。"""
        # 构造候选集合：同会话后续层 + 目录最近文件（去除当前）
        candidates = []
        candidates.extend(self._generate_session_layer_files(accessed_file))
        candidates.extend(self._generate_recent_files(os.path.dirname(accessed_file), exclude=accessed_file))

        # 过滤：已在内存缓存的直接忽略；SSD命中但内存未命中的放前半区（便于“本地SSD写前面的缓存层”）
        filtered: List[str] = []
        ssd_first: List[str] = []
        for k in candidates:
            if self._use_mem and self.mem_cache.exists(k):
                continue
            if self._use_ssd and self.ssd_cache.exists(k):
                ssd_first.append(k)
            else:
                filtered.append(k)

        # 预算估计：用本次加载时间估算可用带宽覆盖的字节量
        budget_bytes = int(self._estimator.budget(front_load_time_sec) * 0.6)
        # 简单切半：一半预算用于SSD→内存的快速回填，另一半用于远端→内存
        half_budget = max(budget_bytes // 2, 0)
        plan = []
        if ssd_first:
            plan.extend(self._planner.build(ssd_first, half_budget))
        if filtered:
            plan.extend(self._planner.build(filtered, budget_bytes - half_budget))
        if not plan:
            return
        # 预先在 buffer 中标记，避免重复提交
        to_submit: List[str] = []
        for key in plan:
            if self._buffer.reserve(key):
                self._buffer.mark_fetching(key)
                to_submit.append(key)
        if to_submit and self._aggregator is not None:
            self._aggregator.submit(to_submit)
    
    def _on_prefetch_ready(self, key: str) -> None:
        try:
            self._buffer.mark_ready(key)
        except Exception:
            pass
    
    def _fetch_to_cache(self, file_path: str) -> Optional[bytes]:
        """由聚合器调用：优先从SSD读入内存；若SSD无，则从后端拉取，再写入内存和SSD（若启用）。"""
        try:
            # SSD → 内存
            if self._use_ssd and self.ssd_cache.exists(file_path):
                data = self.ssd_cache.get(file_path)
                if data is not None and self._use_mem:
                    self.mem_cache.put(file_path, data)
                return data
            # 远端 → 内存 (+SSD)
            data = self.storage_backend.download(file_path)
            if data is not None:
                if self._use_ssd:
                    self.ssd_cache.put(file_path, data)
                if self._use_mem:
                    self.mem_cache.put(file_path, data)
            return data
        except Exception as e:
            logger.error(f"Aggregated prefetch failed for {file_path}: {e}")
            return None
    
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
        # 预取接下来几层的文件（窗口=5层，最大到28层）
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
    
    # NOTE: legacy per-file prefetch removed in favor of IOAggregator