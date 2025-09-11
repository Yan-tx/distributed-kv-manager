import os
import torch
import logging
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from .base import DistributedKVEngineBase, StoreStatus, RetrieveStatus
from distributed_kv_manager.metadata.etcd import KVMetadataManager, KVMetadata
from distributed_kv_manager.metadata.metadata_cache import MetadataCache
from distributed_kv_manager.storage.factory import StorageFactory
from distributed_kv_manager.storage.base import AbstractStorage


# ------------------ Logger 配置 ------------------ #
logger = logging.getLogger("KVEngine")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

class KVEngine(DistributedKVEngineBase):
    """
    分布式 KV 引擎，封装KV存取。
    """

    def __init__(self, config):
        self.rank = getattr(config, "rank", 0)
        self.local_rank = getattr(config, "local_rank", 0)
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._futures = []

        # 使用存储工厂创建存储实例
        self._storage: AbstractStorage = StorageFactory.create_storage(config)
        self.storage_dir = getattr(config.kv_transfer_config, "storage_dir", "/kvcache")

        # metadata
        endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
        self._meta = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")

        # metadata cache
        self._meta_cache = MetadataCache(meta_manager=self._meta)

    # ------------------ KV 存储/检索 ------------------ #
    def should_store(self, model_input) -> StoreStatus:
        # 简单策略：总是存
        return StoreStatus.STORED

    def should_retrieve(self, model_input) -> RetrieveStatus:
        """判断 KV 是否命中缓存"""
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            file_path = self._make_key(current_tokens)

            # 从 MetadataCache 访问元数据
            meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=getattr(model_input, "layer_id", None),
                session_id=getattr(model_input, "session_id", None),
            )

            logger.debug(f"检查序列 {file_path}: meta={meta}, status={getattr(meta, 'status', None) if meta else None}")

            if meta is None or meta.status != 1:
                # 元数据不存在或者状态不是已提交，则 MISS
                logger.debug(f"序列 {file_path} 未命中缓存")
                return RetrieveStatus.MISS

        # 所有序列都存在且已提交
        logger.debug(f"KV Cache 命中")
        return RetrieveStatus.HIT

    def store_kv(self, model_config, parallel_config, transfer_config,
                model_executable, model_input, kv_caches, store_status):
        """基于元数据缓存的两阶段提交写入 KV 缓存和隐藏状态（异步提交）"""
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping
        num_layers = len(kv_caches)

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            file_path = self._make_key(current_tokens)

            # ------------------ 第一步：查询元数据缓存 ------------------ #
            existing_meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=getattr(model_input, "layer_id", None),
                session_id=getattr(model_input, "session_id", None),
            )
            if existing_meta is not None and existing_meta.status == 1:
                logger.debug(f"序列 {file_path} 已存在缓存（通过元数据缓存确认），跳过存储")
                continue

            # ------------------ 第二步：写入元数据缓存（锁定状态） ------------------ #
            meta = KVMetadata(
                session_id=b"session_0000",  # 可根据实际生成
                layer_id=0,                  # 可以按 layer 细化
                token_idx=f"{start_pos}-{end_pos}",
                file_path=file_path,
                file_size=0,                 # 先填0，实际大小写入完成后更新
                create_time=int(time.time()),
                last_access=int(time.time()),
                replica_locations=[b"" for _ in range(3)],
                status=0,                     # 0表示正在写入
                schema_version=1,
                ext_flags=0,
                ext_data=b"",
                ext_data_len=0,
            )
            # 使用缓存 put，自动写入 Pool1/2/3 及 etcd
            self._meta_cache.put_metadata(meta)

            # ------------------ 第三步：准备 KV 数据 ------------------ #
            all_keys, all_values = [], []
            for layer_idx in range(num_layers):
                kv_cache = kv_caches[layer_idx]
                key_cache, value_cache = self.split_kv_cache(kv_cache)
                current_slots = slot_mapping[start_pos:end_pos].flatten()
                all_keys.append(key_cache[current_slots].unsqueeze(0))
                all_values.append(value_cache[current_slots].unsqueeze(0))

            # 确保 all_keys 和 all_values 是张量
            all_keys = torch.cat(all_keys, dim=0) if all_keys else torch.tensor([])
            all_values = torch.cat(all_values, dim=0) if all_values else torch.tensor([])
            
            roi = torch.ones_like(current_tokens, dtype=torch.bool)
            hidden_state = getattr(model_input, "hidden_states", None)
            if hidden_state is not None:
                hidden_state = hidden_state[start_pos:end_pos]

            # ------------------ 第四步：异步写入 KV 和更新元数据 ------------------ #
            def insert_and_update(seq_idx=seq_idx, file_path=file_path, all_keys=all_keys, 
                            all_values=all_values, hidden_state=hidden_state, 
                            current_tokens=current_tokens, roi=roi, meta=meta):
                try:
                    logger.debug(f"开始处理序列 {seq_idx}: {file_path}")
                    
                    # 写入存储
                    self._storage_insert(file_path, all_keys, all_values, hidden_state, current_tokens, roi)
                    logger.debug(f"序列 {seq_idx} 存储完成")
                    
                    # 写入完成后更新元数据
                    keys_size = all_keys.numel() * all_keys.element_size() if all_keys.numel() > 0 else 0
                    values_size = all_values.numel() * all_values.element_size() if all_values.numel() > 0 else 0
                    meta.file_size = keys_size + values_size
                    meta.status = 1  # 写入完成
                    meta.last_access = int(time.time())
                    
                    # 使用缓存 put 更新元数据
                    self._meta_cache.put_metadata(meta)
                    logger.debug(f"序列 {seq_idx} 元数据更新完成，状态: {meta.status}")
                    
                except Exception as e:
                    logger.error(f"存储序列 {seq_idx} ({file_path}) 失败: {e}", exc_info=True)
            
            future = self._executor.submit(insert_and_update)
            self._futures.append(future)

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status):
        """
        从 KV 缓存检索 KV 与隐藏状态。
        bypass_model_exec 由外部传入的 retrieve_status 决定。
        """
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        num_layers = len(kv_caches)

        all_hidden_states = []
        bypass_model_exec = (retrieve_status == RetrieveStatus.HIT)

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len

            if start_pos >= num_prefill_tokens:
                bypass_model_exec = False
                continue

            current_tokens = input_tokens[start_pos:end_pos]
            file_path = self._make_key(current_tokens)
            roi = torch.ones_like(current_tokens, dtype=torch.bool)

            if not bypass_model_exec:
                continue  # 没命中，直接交给模型执行

            # 元数据命中，则读取 KV
            key, value, hidden = self._storage_download(file_path)
            if key is None or value is None or hidden is None:
                bypass_model_exec = False
                continue

            device = input_tokens.device
            key, value, hidden = key.to(device), value.to(device), hidden.to(device)

            current_slots = slot_mapping[start_pos:end_pos]
            for layer_idx in range(num_layers):
                layer_k, layer_v = key[layer_idx], value[layer_idx]
                kv_cache = kv_caches[layer_idx]
                key_cache, value_cache = self.split_kv_cache(kv_cache)
                key_cache[current_slots] = layer_k
                value_cache[current_slots] = layer_v
                kv_caches[layer_idx] = torch.stack([key_cache, value_cache], dim=0)

            all_hidden_states.append(hidden)

        if bypass_model_exec and all_hidden_states:
            final_hidden_state = torch.cat(all_hidden_states, dim=0)
            return final_hidden_state, True, model_input
        else:
            return None, False, model_input

    # ------------------ Helper: KV 结构 ------------------ #
    @staticmethod
    def split_kv_cache(kv_cache: torch.Tensor):
        if kv_cache.dim() < 2 or kv_cache.size(0) != 2:
            raise ValueError(f"无法识别的KV缓存结构: {kv_cache.shape}")
        return kv_cache[0], kv_cache[1]

    # ------------------ Helper: 文件键 ------------------ #
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """为张量生成唯一哈希键"""
        if tensor.numel() == 0:
            return "empty"
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.blake2b(tensor_bytes).hexdigest()

    def _make_key(self, input_tokens: torch.Tensor) -> str:
        seq_hash = self._tensor_hash(input_tokens)
        return os.path.join(self.storage_dir, f"kv_{seq_hash}.pt")
    
    def _storage_insert(self, file_path: str, k_cache, v_cache, hidden, input_tokens, roi):
        """使用存储后端打包并上传数据"""
        data = self._storage.pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi)
        success = self._storage.upload(file_path, data)
        if not success:
            logger.error(f"Failed to upload KV data to {file_path}")

    def _storage_download(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """使用存储后端下载并解包数据"""
        data = self._storage.download(file_path)
        if data is None:
            return None, None, None
        return self._storage.unpack_kv_data(data)

    def _storage_exists(self, file_path: str) -> bool:
        """使用存储后端检查文件是否存在"""
        return self._storage.exists(file_path)

    def close(self):
        logger.debug(f"等待 {len(self._futures)} 个异步任务完成")
        
        for i, f in enumerate(self._futures):
            try:
                # 设置更长的超时时间
                f.result(timeout=120)
                logger.debug(f"异步任务 {i} 完成")
            except Exception as e:
                logger.error(f"异步任务 {i} 失败: {e}", exc_info=True)
        
        self._executor.shutdown(wait=True)
        logger.debug("所有异步任务处理完成")
        print("[KV ENGINE CLOSED]")

# --- module-level engine wrapper ---
from .base import StoreStatus, RetrieveStatus

_engine_singleton: KVEngine | None = None  

def init_engine(config):
    """
    初始化并返回 engine 单例。传入 vllm 的 config。
    """
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = KVEngine(config)  # 创建KVEngine实例
    return _engine_singleton

def destroy_engine():
    """销毁 engine（如果存在）"""
    global _engine_singleton
    if _engine_singleton is not None:
        try:
            _engine_singleton.close()
        finally:
            _engine_singleton = None

def should_store(model_input):
    """
    模块级 should_store 接口，返回 StoreStatus。
    """
    engine = _engine_singleton
    if engine is None:
        # 如果尚未初始化，默认返回 STORED
        return StoreStatus.STORED
    return engine.should_store(model_input)

def store_kv(model_config, parallel_config, transfer_config,
             model_executable, model_input, kv_caches, store_status):
    """
    模块级 store_kv 接口，委托给 engine 的 store_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.store_kv(model_config, parallel_config, transfer_config,
                           model_executable, model_input, kv_caches, store_status)

def should_retrieve(model_input):
    """
    模块级 should_retrieve 接口，返回 RetrieveStatus。
    """
    engine = _engine_singleton
    if engine is None:
        return RetrieveStatus.MISS
    return engine.should_retrieve(model_input)

def retrieve_kv(model_executable, model_input, kv_caches, retrieve_status):
    """
    模块级 retrieve_kv 接口，委托给 engine 的 retrieve_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.retrieve_kv(model_executable, model_input, kv_caches, retrieve_status)