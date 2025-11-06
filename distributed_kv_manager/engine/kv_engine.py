import os
import torch
import logging
import hashlib
import time
import struct
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from types import SimpleNamespace
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
        _storage_inst = StorageFactory.create_storage(config)
        if _storage_inst is None:
            raise RuntimeError("StorageFactory.create_storage 返回 None")
        self._storage = _storage_inst
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

    def store_kv(
        self,
        model_config,
        parallel_config,
        transfer_config,
        model_executable,
        model_input,
        kv_caches,
        store_status,
        hidden_or_intermediate_states=None,
        ):
        """基于元数据缓存的两阶段提交写入 KV 缓存（异步提交）。

        Args:
            hidden_or_intermediate_states: 兼容旧接口的占位参数（当前未使用）。
        """
        _ = hidden_or_intermediate_states  # 占位以保持接口兼容
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
            # 当前序列对应的槽位映射
            current_slots = slot_mapping[seq_idx]
            if current_slots.dim() > 1:
                current_slots = current_slots.reshape(-1)
            if current_slots.dim() == 0:
                current_slots = current_slots.unsqueeze(0)
            if current_slots.numel() != seq_len:
                logger.warning(
                    "序列 %s 的槽位映射长度(%d)与序列长度(%d)不一致，使用最小长度", 
                    file_path,
                    current_slots.numel(),
                    seq_len,
                )
                limit = min(current_slots.numel(), seq_len)
                current_slots = current_slots[:limit]
                current_tokens = current_tokens[:limit]
                end_pos = start_pos + limit
                seq_len = limit

            for layer_idx in range(num_layers):
                kv_cache = kv_caches[layer_idx]
                key_cache, value_cache = self.split_kv_cache(kv_cache)

                # 仅存储当前序列对应的 KV
                all_keys.append(key_cache[current_slots].unsqueeze(0))
                all_values.append(value_cache[current_slots].unsqueeze(0))

            all_keys = torch.cat(all_keys, dim=0) if all_keys else torch.tensor([])
            all_values = torch.cat(all_values, dim=0) if all_values else torch.tensor([])
            
            roi = torch.ones_like(current_tokens, dtype=torch.bool)

            # ------------------ 第四步：异步写入 KV 和更新元数据 ------------------ #
            # 使用辅助类确保变量正确捕获
            class InsertAndUpdateTask:
                def __init__(
                    self,
                    engine,
                    seq_idx,
                    file_path,
                    all_keys,
                    all_values,
                    current_tokens,
                    roi,
                    current_slots,
                    meta,
                ):
                    self.engine = engine
                    self.seq_idx = seq_idx
                    self.file_path = file_path
                    self.all_keys = all_keys
                    self.all_values = all_values
                    self.current_tokens = current_tokens
                    self.roi = roi
                    self.current_slots = current_slots
                    self.meta = meta
                
                def __call__(self):
                    try:
                        logger.debug(f"开始处理序列 {self.seq_idx}: {self.file_path}")
                        
                        # 构造 payload_meta
                        payload_meta = {
                            "schema_version": 1,
                            "tokens_hash": self.engine._tensor_hash(self.current_tokens),
                            "num_layers": int(self.all_keys.shape[0]) if self.all_keys.numel() > 0 else 0,
                            "kv_dtype": str(self.all_keys.dtype) if self.all_keys.numel() > 0 else "unknown",
                            "kv_tail_shape": list(self.all_keys.shape[2:]) if self.all_keys.numel() > 0 else [],
                        }

                        # 写入存储（包含slot_mapping与payload_meta）
                        self.engine._storage_insert(
                            self.file_path,
                            self.all_keys,
                            self.all_values,
                            self.current_tokens,
                            self.roi,
                            self.current_slots,
                            payload_meta,
                        )
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
            task = InsertAndUpdateTask(
                self,
                seq_idx,
                file_path,
                all_keys,
                all_values,
                current_tokens,
                roi,
                current_slots,
                meta,
            )
            future = self._executor.submit(task)
            self._futures.append(future)

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status):
        """从存储中恢复 KV 缓存，并按照 LMCache 的返回约定提供结果。"""
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping  # 保持二维形状
        num_layers = len(kv_caches)

        if retrieve_status != RetrieveStatus.HIT:
            logger.debug("检索状态为 MISS，跳过 KV 恢复")
            return None, False, model_input

        logger.debug(f"开始检索KV，序列数量: {len(seq_lens)}")

        # 标记全部序列是否都成功恢复（用于决定是否 bypass 前向）
        recovered_sequences = 0

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len

            current_tokens = input_tokens[start_pos:end_pos]
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            file_path = self._make_key(current_tokens, session_id, layer_id)

            meta = self._meta_cache.get_metadata(key=file_path)
            if meta and meta.is_expired():
                logger.debug(f"序列 {seq_idx} 元数据已过期: {file_path}")
                continue

            self._update_last_access_time(file_path)

            # 读取纯KV负载与扩展信息
            kv_bytes = self._storage_download_kv_bytes(file_path)
            if kv_bytes is None:
                logger.debug(f"序列 {seq_idx} 的 KV 数据缺失或损坏: {file_path}")
                continue

            payload_info = self._storage.extract_payload_info(kv_bytes)

            # 优先使用持久化的slot_mapping（若存在）
            persisted_slots = payload_info.get("slot_mapping")

            # 可选一致性校验：token哈希/层数/形状
            payload_meta = payload_info.get("payload_meta", {}) if isinstance(payload_info, dict) else {}
            expected_layers = payload_meta.get("num_layers")
            tokens_hash = payload_meta.get("tokens_hash")

            try:
                # 获取 KV 张量
                key, value = self._storage.unpack_kv_data(kv_bytes)
            except Exception:
                key, value = None, None
            if key is None or value is None:
                logger.debug(f"序列 {seq_idx} 的 KV 数据解包失败: {file_path}")
                continue

            device = input_tokens.device
            key = key.to(device)
            value = value.to(device)

            current_slots = slot_mapping[seq_idx]
            if current_slots.dim() > 1:
                current_slots = current_slots.reshape(-1)
            if current_slots.dim() == 0:
                current_slots = current_slots.unsqueeze(0)
            if current_slots.numel() != seq_len:
                logger.warning(
                    "序列 %s 的槽位映射长度(%d)与序列长度(%d)不一致",
                    file_path,
                    current_slots.numel(),
                    seq_len,
                )
                limit = min(current_slots.numel(), seq_len)
                current_slots = current_slots[:limit]
                seq_len = limit
                end_pos = start_pos + limit
                if limit == 0:
                    continue

            # 若存在持久化的 slot_mapping，则优先使用
            if persisted_slots is not None:
                try:
                    persisted_slots = persisted_slots.reshape(-1)
                    if persisted_slots.numel() != current_slots.numel():
                        limit = min(persisted_slots.numel(), current_slots.numel())
                        persisted_slots = persisted_slots[:limit]
                        current_slots = current_slots[:limit]
                        seq_len = limit
                        if limit == 0:
                            continue
                    slots_to_use = persisted_slots.to(current_slots.device)
                except Exception:
                    slots_to_use = current_slots
            else:
                slots_to_use = current_slots

            # 轻量校验：层数与token哈希（存在时）
            if expected_layers is not None and int(expected_layers) != len(kv_caches):
                logger.warning(
                    "层数不一致: payload=%s, runtime=%s; 跳过该序列",
                    expected_layers,
                    len(kv_caches),
                )
                continue
            if tokens_hash is not None:
                cur_hash = self._tensor_hash(current_tokens)
                if tokens_hash != cur_hash:
                    logger.warning("token哈希不一致，跳过该序列: %s", file_path)
                    continue

            logger.debug(
                "恢复 KV: 序列 %d, file=%s, 槽位数=%d",
                seq_idx,
                file_path,
                current_slots.numel(),
            )

            try:
                for layer_idx in range(num_layers):
                    layer_k, layer_v = key[layer_idx], value[layer_idx]
                    kv_cache = kv_caches[layer_idx]
                    key_cache, value_cache = self.split_kv_cache(kv_cache)

                    if layer_k.shape[0] != slots_to_use.numel():
                        logger.error(
                            "KV 形状不匹配: 层 %d 期望 %d 个token, 实际 %d",
                            layer_idx,
                            slots_to_use.numel(),
                            layer_k.shape[0],
                        )
                        raise ValueError("KV shape mismatch")

                    key_cache[slots_to_use] = layer_k.to(key_cache.dtype)
                    value_cache[slots_to_use] = layer_v.to(value_cache.dtype)

                recovered_sequences += 1
            except Exception as e:
                logger.error(
                    f"在序列 {seq_idx} 层 {layer_idx} 恢复 KV 时出错: {e}",
                    exc_info=True,
                )
                break

        if recovered_sequences != len(seq_lens):
            logger.debug(
                "KV 恢复未全部成功 (成功 %d / 期望 %d)，继续执行模型前向",
                recovered_sequences,
                len(seq_lens),
            )
            return None, False, model_input

        # 全量命中：构造全 0 hidden 占位以允许上层绕过模型前向
        try:
            total_tokens = sum(seq_lens)
            hidden_dim = None
            # 优先从 model_executable 中推断 embedding 维度
            if model_executable is not None:
                emb_mod = getattr(getattr(model_executable, "model", None), "embed_tokens", None)
                if emb_mod is not None:
                    hidden_dim = getattr(emb_mod, "embedding_dim", None)
                    if hidden_dim is None and hasattr(emb_mod, "weight"):
                        # 退化：从权重形状猜测 (vocab, hidden_dim)
                        wt = getattr(emb_mod, "weight")
                        if hasattr(wt, "shape") and len(wt.shape) >= 2:
                            hidden_dim = wt.shape[-1]
            # 如果仍无法获取，使用 KV 的尾部维度近似（num_heads * head_dim）
            if hidden_dim is None:
                # 期望 kv_caches[layer] 形状: [2, total_slots, num_heads, head_dim] 或更多尾部维
                sample = kv_caches[0]
                if sample.dim() >= 4:
                    hidden_dim = sample.shape[-1] * sample.shape[-2]
                else:
                    hidden_dim = 1  # 最小占位
            device = input_tokens.device
            dtype = kv_caches[0].dtype if len(kv_caches) > 0 else torch.float16
            hidden_placeholder = torch.zeros(
                (total_tokens, hidden_dim), device=device, dtype=dtype
            )
        except Exception as e:
            logger.warning(f"构造隐藏占位失败，fallback 为 None: {e}")
            hidden_placeholder = None
            return hidden_placeholder, False, model_input

        logger.debug(
            "KV 全量命中，返回零张量 hidden 占位并设置 bypass=True, shape=%s",
            None if hidden_placeholder is None else tuple(hidden_placeholder.shape),
        )
        return hidden_placeholder, True, model_input

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

    def _make_key(
        self,
        input_tokens: torch.Tensor,
        session_id: Optional[bytes] = None,
        layer_id: Optional[int] = None,
    ) -> str:
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
    
    def _storage_insert(
        self,
        file_path: str,
        k_cache,
        v_cache,
        input_tokens,
        roi,
        slot_mapping: Optional[torch.Tensor] = None,
        payload_meta: Optional[dict] = None,
    ):
        """使用存储后端打包并上传数据，并嵌入元数据用于恢复（包含slot_mapping与meta）。"""
        # 使用扩展打包完整负载
        data = self._storage.pack_full_payload(
            k_cache, v_cache, input_tokens, roi, slot_mapping, payload_meta
        )

        # 获取文件的元数据并嵌入（如可用）
        meta = self._meta_cache.get_metadata(key=file_path)
        if meta:
            metadata_bytes = meta.pack_with_embedding()
            data = metadata_bytes + data

        success = self._storage.upload(file_path, data)
        if not success:
            logger.error(f"Failed to upload KV data to {file_path}")

    def _storage_download(
        self, file_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """使用存储后端下载并解包数据"""
        logger.debug(f"开始下载文件: {file_path}")
        data = self._storage.download(file_path)
        if data is None:
            logger.warning(f"下载文件 {file_path} 失败，文件可能不存在")
            return None, None
            
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

    def _storage_download_kv_bytes(self, file_path: str) -> Optional[bytes]:
        """下载文件并返回纯KV负载字节（去掉嵌入的元数据头）。"""
        data = self._storage.download(file_path)
        if data is None:
            return None
        try:
            from distributed_kv_manager.metadata.etcd import (
                METADATA_HEADER,
                METADATA_VERSION,
                HEADER_SIZE,
                METADATA_SIZE,
            )
            if len(data) >= HEADER_SIZE + METADATA_SIZE:
                header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
                if header == METADATA_HEADER and version == METADATA_VERSION:
                    return data[HEADER_SIZE + METADATA_SIZE :]
        except Exception:
            pass
        return data

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

def init_engine(config=None, config_path: Optional[str] = None):
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
            config = load_config_from_json(config_path) if config_path is not None else load_config_from_json()
        else:
            # 尝试加载config.json，并与传入的config合并，使JSON生效
            json_cfg = None
            try:
                # 优先使用显式提供的config_path，否则使用默认查找
                json_cfg = load_config_from_json(config_path) if config_path is not None else load_config_from_json()
            except FileNotFoundError:
                # 若找不到则保持传入的config
                json_cfg = None

            if json_cfg is not None:
                combined = SimpleNamespace()
                # 优先使用传入config的rank/local_rank，其次使用JSON，最后默认0
                combined.rank = getattr(config, "rank", getattr(json_cfg, "rank", 0))
                combined.local_rank = getattr(config, "local_rank", getattr(json_cfg, "local_rank", 0))
                # 引擎标识（如存在）
                if hasattr(config, "engine_id"):
                    combined.engine_id = getattr(config, "engine_id")
                elif hasattr(json_cfg, "engine_id"):
                    combined.engine_id = getattr(json_cfg, "engine_id")
                # 使用JSON中的kv_transfer_config覆盖，以确保config.json生效
                combined.kv_transfer_config = getattr(
                    json_cfg, "kv_transfer_config",
                    getattr(config, "kv_transfer_config", SimpleNamespace())
                )
                config = combined

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
             model_executable, model_input, kv_caches, store_status,
             hidden_or_intermediate_states=None):
    """
    模块级 store_kv 接口，委托给 engine 的 store_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.store_kv(model_config, parallel_config, transfer_config,
                           model_executable, model_input, kv_caches, store_status,
                           hidden_or_intermediate_states)

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
