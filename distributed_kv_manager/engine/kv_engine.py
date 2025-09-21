import os
import torch
import logging
import hashlib
import time
import struct
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union
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
        logger.info(f"创建的存储实例类型: {type(self._storage)}")
        self.storage_dir = getattr(config.kv_transfer_config, "storage_dir", "/kvcache")

        # metadata
        endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
        self._meta = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")

        # metadata cache
        self._meta_cache = MetadataCache(meta_manager=self._meta)
        
        # cleanup manager
        # 读取配置的清理间隔时间，默认1小时
        cleanup_interval = getattr(config.kv_transfer_config, "cleanup_interval", 3600)
        from distributed_kv_manager.metadata.cleanup import KVCleanupManager
        self._cleanup_manager = KVCleanupManager(self._meta, cleanup_interval, self._storage)
        self._cleanup_manager.start()

    # ------------------ KV 存储/检索 ------------------ #
    def should_store(self, model_input) -> StoreStatus:
        # 简单策略：总是存
        return StoreStatus.STORED

    def should_retrieve(self, model_input) -> RetrieveStatus:
        """
        判断 KV 是否命中缓存，通过元数据缓存确认：
        - status == 1 表示写入完成，可以命中
        - 其他情况视为 MISS
        """
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            # 获取session_id和layer_id
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            file_path = self._make_key(current_tokens, session_id, layer_id)

            # 从 MetadataCache 访问元数据
            meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=layer_id,
                session_id=session_id,
            )

            if meta is None or meta.status != 1:
                # 元数据不存在或者状态不是已提交，则 MISS
                return RetrieveStatus.MISS
                
            # 检查元数据是否已过期
            if meta.is_expired():
                # 元数据已过期，则 MISS
                logger.debug(f"元数据已过期: {file_path}")
                return RetrieveStatus.MISS

        # 所有序列都存在且已提交
        logger.debug(f"KV Cache 命中")
        return RetrieveStatus.HIT

    def store_kv(self, model_config, parallel_config, transfer_config,
            model_executable, model_input, kv_caches, store_status, hidden_or_intermediate_states):
        """基于元数据缓存的两阶段提交写入 KV 缓存和隐藏状态（异步提交）"""
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping  # 保持二维形状
        num_layers = len(kv_caches)

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            # 获取session_id和layer_id
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            file_path = self._make_key(current_tokens, session_id, layer_id)

            # ------------------ 第一步：查询元数据缓存 ------------------ #
            existing_meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=layer_id,
                session_id=session_id,
            )
            if existing_meta is not None and existing_meta.status == 1:
                logger.debug(f"序列 {file_path} 已存在缓存（通过元数据缓存确认），跳过存储")
                continue

            # ------------------ 第二步：写入元数据缓存（锁定状态） ------------------ #
            # 读取配置的过期时间，默认1天
            expire_time = getattr(self.config.kv_transfer_config, "kv_expire_time", 86400)
            
            meta = KVMetadata(
                session_id=session_id if session_id is not None else b"session_0000",  # 使用实际的session_id或默认值
                layer_id=layer_id if layer_id is not None else 0,                  # 使用实际的layer_id或默认值
                token_idx=f"{start_pos}-{end_pos}",
                file_path=file_path,
                file_size=0,                 # 先填0，实际大小写入完成后更新
                create_time=int(time.time()),
                last_access=int(time.time()),
                expire_time=expire_time,     # 使用配置的过期时间
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
                
                # 关键修改：确保只存储当前序列的KV缓存
                # 使用序列索引获取当前序列的槽位映射
                current_slots = slot_mapping[seq_idx].flatten()
                
                # 确保只选择当前序列的KV缓存
                all_keys.append(key_cache[current_slots].unsqueeze(0))
                all_values.append(value_cache[current_slots].unsqueeze(0))

            all_keys = torch.cat(all_keys, dim=0) if all_keys else torch.tensor([])
            all_values = torch.cat(all_values, dim=0) if all_values else torch.tensor([])
            
            roi = torch.ones_like(current_tokens, dtype=torch.bool)
            hidden_state = hidden_or_intermediate_states
            if hidden_state is not None:
                # hidden_state = hidden_state[start_pos:end_pos]
                logger.debug(f"准备存储的hidden_state形状: {hidden_state.shape}")
            else:
                logger.debug("准备存储的hidden_state为None (正常情况，若仅使用KV Cache)")

            # ------------------ 第四步：异步写入 KV 和更新元数据 ------------------ #
            # 使用辅助类确保变量正确捕获
            class InsertAndUpdateTask:
                def __init__(self, engine, seq_idx, file_path, all_keys, all_values, 
                            hidden_state, current_tokens, roi, meta):
                    self.engine = engine
                    self.seq_idx = seq_idx
                    self.file_path = file_path
                    self.all_keys = all_keys
                    self.all_values = all_values
                    self.hidden_state = hidden_state
                    self.current_tokens = current_tokens
                    self.roi = roi
                    self.meta = meta
                
                def __call__(self):
                    try:
                        logger.debug(f"开始处理序列 {self.seq_idx}: {self.file_path}")
                        
                        # 写入存储
                        self.engine._storage_insert(self.file_path, self.all_keys, self.all_values, 
                                                self.hidden_state, self.current_tokens, self.roi)
                        logger.debug(f"序列 {self.seq_idx} 存储完成")
                        
                        # 写入完成后更新元数据
                        keys_size = self.all_keys.numel() * self.all_keys.element_size() if self.all_keys.numel() > 0 else 0
                        values_size = self.all_values.numel() * self.all_values.element_size() if self.all_values.numel() > 0 else 0
                        self.meta.file_size = keys_size + values_size
                        self.meta.status = 1  # 写入完成
                        self.meta.last_access = int(time.time())
                        
                        # 使用缓存 put 更新元数据
                        self.engine._meta_cache.put_metadata(self.meta)
                        logger.debug(f"序列 {self.seq_idx} 元数据更新完成，状态: {self.meta.status}")
                        
                    except Exception as e:
                        logger.error(f"存储序列 {self.seq_idx} ({self.file_path}) 失败: {e}", exc_info=True)

            # 创建任务实例并提交
            task = InsertAndUpdateTask(self, seq_idx, file_path, all_keys, all_values, 
                                    hidden_state, current_tokens, roi, meta)
            future = self._executor.submit(task)
            self._futures.append(future)

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status):
        """
        从 KV 缓存检索 KV 与隐藏状态。
        bypass_model_exec 由外部传入的 retrieve_status 决定。
        """
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping  # 保持二维形状
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        num_layers = len(kv_caches)

        # num_tok = len(model_input.input_tokens)
        # num_dim = model_executable.model.embed_tokens.embedding_dim
        # dtype = model_executable.model.embed_tokens.weight.dtype
        # device = model_input.input_tokens.device
        # all_hidden_states = torch.zeros(
        #     num_tok, num_dim, device=device, dtype=dtype
        # )
        all_hidden_states = []

        bypass_model_exec = (retrieve_status == RetrieveStatus.HIT)

        logger.debug(f"开始检索KV，检索状态: {retrieve_status}, 序列数量: {len(seq_lens)}")
        
        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len

            if start_pos >= num_prefill_tokens:
                bypass_model_exec = False
                continue

            current_tokens = input_tokens[start_pos:end_pos]
            # 获取session_id和layer_id
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            file_path = self._make_key(current_tokens, session_id, layer_id)
            roi = torch.ones_like(current_tokens, dtype=torch.bool)

            if not bypass_model_exec:
                continue  # 没命中，直接交给模型执行

            # 元数据命中，则读取 KV
            logger.debug(f"检索序列 {seq_idx}: {file_path}")
            
            # 检查元数据是否已过期
            meta = self._meta_cache.get_metadata(key=file_path)
            if meta and meta.is_expired():
                logger.warning(f"序列 {seq_idx} 元数据已过期")
                bypass_model_exec = False
                continue
            
            # 更新last_access时间实现续命（异步更新）
            self._update_last_access_time(file_path)
            
            key, value, hidden = self._storage_download(file_path)
            all_hidden_states.append(hidden)
            logger.debug(f"检索到的hidden形状: {hidden.shape if hidden is not None else 'None'}")
            
            if key is None or value is None :
                logger.warning(f"序列 {seq_idx} 检索失败，文件可能不存在或损坏")
                logger.warning(f"文件路径: {file_path}")
                # 检查文件是否存在
                if not self._storage_exists(file_path):
                    logger.warning(f"文件 {file_path} 不存在")
                else:
                    logger.warning(f"文件 {file_path} 存在但可能已损坏")
                    # 增加更多诊断信息
                    logger.warning(f"下载的key是否为None: {key is None}")
                    logger.warning(f"下载的value是否为None: {value is None}")
                bypass_model_exec = False
                continue

            device = input_tokens.device
            key, value = key.to(device), value.to(device)

            # 打印检索到的KV形状信息
            logger.debug(f"检索到的KV形状 - key: {key.shape}, value: {value.shape}, hidden: {hidden.shape if hidden is not None else 'None'}")
            
            # 使用序列索引获取当前序列的槽位映射
            current_slots = slot_mapping[seq_idx]
            logger.debug(f"当前槽位: {current_slots}, 形状: {current_slots.shape}")

            try:
                for layer_idx in range(num_layers):
                    layer_k, layer_v = key[layer_idx], value[layer_idx]
                    kv_cache = kv_caches[layer_idx]
                    key_cache, value_cache = self.split_kv_cache(kv_cache)
                    
                    # 打印形状信息，帮助诊断问题
                    logger.debug(f"层 {layer_idx}: layer_k形状: {layer_k.shape}, layer_v形状: {layer_v.shape}")
                    logger.debug(f"层 {layer_idx}: key_cache形状: {key_cache.shape}, value_cache形状: {value_cache.shape}")
                    logger.debug(f"层 {layer_idx}: 当前槽位形状: {current_slots.shape}")
                    
                    # 检查形状是否匹配
                    if layer_k.shape[0] != current_slots.numel():
                        logger.error(f"形状不匹配: 层 {layer_idx} 的KV缓存有 {layer_k.shape[0]} 个token, 但槽位映射有 {current_slots.numel()} 个槽位")
                        bypass_model_exec = False
                        break
                        
                    # 如果 current_slots 是标量，需要特殊处理
                    if current_slots.dim() == 0:
                        current_slots = current_slots.unsqueeze(0)
                    
                    # 直接更新 key_cache 和 value_cache，避免使用 torch.stack 创建临时大张量
                    # torch.stack 会创建一个新的张量，可能导致内存峰值
                    key_cache[current_slots] = layer_k
                    value_cache[current_slots] = layer_v
                    # 分别更新 kv_cache 的两个部分
                    kv_cache[0] = key_cache
                    kv_cache[1] = value_cache
                    # 不再使用 torch.stack，因为 kv_cache 本身已经是正确的结构
                    # kv_caches[layer_idx] = torch.stack([key_cache, value_cache], dim=0)
                    
            except Exception as e:
                logger.error(f"在序列 {seq_idx} 层 {layer_idx} 处理KV缓存时发生错误: {e}")
                logger.error(f"key_cache形状: {key_cache.shape}, layer_k形状: {layer_k.shape}")
                logger.error(f"current_slots: {current_slots}")
                bypass_model_exec = False
                # 重新抛出异常，以便外部捕获
                raise

        if bypass_model_exec and len(all_hidden_states) > 0:
            return all_hidden_states, True, model_input
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

    def _make_key(self, input_tokens: torch.Tensor, session_id: bytes = None, layer_id: int = None) -> str:
        seq_hash = self._tensor_hash(input_tokens)
        # 如果没有提供session_id和layer_id，使用默认值
        if session_id is None:
            session_id = b"session_0000"
        if layer_id is None:
            layer_id = 0
        # 将session_id转换为字符串
        session_str = session_id.decode('utf-8') if isinstance(session_id, bytes) else str(session_id)
        # 构造文件名，不包含路径前缀
        return f"kv_{session_str}_layer_{layer_id}_{seq_hash}.pt"
    
    def _storage_insert(self, file_path: str, k_cache, v_cache, hidden, input_tokens, roi):
        """使用存储后端打包并上传数据，并嵌入元数据用于恢复"""
        # 首先打包KV数据
        data = self._storage.pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi)
        
        # 获取文件的元数据
        meta = self._meta_cache.get_metadata(key=file_path)
        if meta:
            # 将元数据打包并嵌入到数据开头
            metadata_bytes = meta.pack_with_embedding()
            data = metadata_bytes + data
            
        success = self._storage.upload(file_path, data)
        if not success:
            logger.error(f"Failed to upload KV data to {file_path}")

    def _storage_download(self, file_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """使用存储后端下载并解包数据"""
        logger.debug(f"开始下载文件: {file_path}")
        data = self._storage.download(file_path)
        if data is None:
            logger.warning(f"下载文件 {file_path} 失败，文件可能不存在")
            return None, None, None
            
        logger.debug(f"成功下载文件 {file_path}，数据大小: {len(data)} 字节")
            
        # 检查数据是否包含嵌入的元数据
        # 元数据头部是METADATA_HEADER + METADATA_VERSION
        from distributed_kv_manager.metadata.etcd import METADATA_HEADER, METADATA_VERSION, HEADER_SIZE, METADATA_SIZE
        header_size = HEADER_SIZE  # 8 bytes
        min_data_size = header_size + METADATA_SIZE  # 至少需要头部和元数据大小
        
        # 如果数据足够长，检查是否有元数据头部
        if len(data) >= min_data_size:
            try:
                # 提取头部
                header, version = struct.unpack("<4sI", data[:header_size])
                logger.debug(f"文件 {file_path} 包含元数据头部: {header}, 版本: {version}")
                # 如果有元数据头部，跳过元数据部分
                if header == METADATA_HEADER and version == METADATA_VERSION:
                    # 跳过头部和元数据，只解包KV数据
                    kv_data = data[header_size + METADATA_SIZE:]
                    logger.debug(f"跳过元数据部分，KV数据大小: {len(kv_data)} 字节")
                    return self._storage.unpack_kv_data(kv_data)
                else:
                    logger.warning(f"文件 {file_path} 的元数据头部不匹配: {header} != {METADATA_HEADER} 或版本不匹配: {version} != {METADATA_VERSION}")
            except Exception as e:
                logger.warning(f"从文件 {file_path} 中提取元数据失败: {e}")
        else:
            logger.warning(f"文件 {file_path} 数据太短，无法包含元数据: {len(data)} < {min_data_size}")
        
        # 如果没有元数据嵌入或提取失败，直接解包整个数据
        logger.debug(f"直接解包整个数据，大小: {len(data)} 字节")
        return self._storage.unpack_kv_data(data)

    def _storage_exists(self, file_path: str) -> bool:
        """使用存储后端检查文件是否存在"""
        logger.debug(f"检查文件是否存在: {file_path}, 存储实例类型: {type(self._storage)}")
        # 直接使用file_path，因为_storage.exists会处理路径前缀
        exists = self._storage.exists(file_path)
        logger.debug(f"文件 {file_path} 存在: {exists}")
        return exists
        
    def _update_last_access_time(self, file_path: str):
        """更新元数据的最后访问时间（异步）"""
        try:
            # 获取当前元数据
            meta = self._meta_cache.get_metadata(key=file_path)
            if meta:
                # 更新最后访问时间
                meta.last_access = int(time.time())
                # 异步更新到缓存和ETCD
                self._meta_cache.put_metadata(meta)
        except Exception as e:
            logger.error(f"更新元数据最后访问时间失败: {e}")

    def close(self):
        """关闭KV引擎，等待所有异步任务完成"""
        # 关闭清理器
        if hasattr(self, '_cleanup_manager'):
            self._cleanup_manager.stop()
        
        # 等待异步任务完成
        for f in self._futures:
            try:
                f.result(timeout=60)
            except Exception as e:
                logger.error(f"KV 异步存储失败: {e}")
        self._executor.shutdown(wait=True)
        print("[KV ENGINE CLOSED]")

# --- module-level engine wrapper ---
from .base import StoreStatus, RetrieveStatus
from ..config_loader import load_config_from_json

_engine_singleton: KVEngine | None = None  

def init_engine(config=None, config_path=None):
    """
    初始化并返回 engine 单例。传入 vllm 的 config 或从 config.json 读取配置。
    
    Args:
        config: vllm 的配置对象
        config_path: 配置文件路径，默认为项目根目录的config.json
    """
    global _engine_singleton
    if _engine_singleton is None:
        # 如果没有提供config，则从配置文件加载
        if config is None:
            config = load_config_from_json(config_path)
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
             model_executable, model_input, kv_caches, store_status, hidden_or_intermediate_states):
    """
    模块级 store_kv 接口，委托给 engine 的 store_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.store_kv(model_config, parallel_config, transfer_config,
                           model_executable, model_input, kv_caches, store_status, hidden_or_intermediate_states)

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