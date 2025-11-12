from __future__ import annotations

import logging
from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Set

import torch

# v1 base (provided in this package)
from .base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

# Reuse storage/metadata logic from the existing engine module-level API
from distributed_kv_manager.config_loader import load_config_from_json
from distributed_kv_manager.storage.v1.storage import V1Storage, create_v1_storage
from distributed_kv_manager.storage.v1.layered_storage import LayeredV1Storage
from distributed_kv_manager.metadata.v1.metadata import V1MetadataClient
from distributed_kv_manager.metadata.etcd import KVMetadata, KVMetadataManager

try:
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore
    logger = _vllm_init_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


class _ReqLoadSpec:
    def __init__(self, vllm_cached_tokens: int, cached_tokens: int, can_load: bool):
        self.vllm_cached_tokens = vllm_cached_tokens
        self.cached_tokens = cached_tokens
        self.can_load = can_load


@dataclass
class _LoadSpan:
    start: int
    end: int


@dataclass
class _StoreSpan:
    start: int
    end: int


@dataclass
class _RequestPlan:
    req_id: str
    session_id: str
    token_ids: List[int]
    slot_mapping: List[int]
    load_spans: List[_LoadSpan] = field(default_factory=list)
    store_spans: List[_StoreSpan] = field(default_factory=list)
    is_last_prefill: bool = False


class _V1Metadata(KVConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: List[_RequestPlan] = []

    def add_request(self, request_plan: _RequestPlan) -> None:
        self.requests.append(request_plan)


class V1KVEngineImpl(KVConnectorBase_V1):
    """Implement v1 connector接口，但逻辑驻留在 engine/v1 层。

    - Worker 侧：接收 forward_context/kv_caches，调用底层引擎存取。
    - Scheduler 侧：最小实现，当前不做异步传输计划。
    - 存取格式与 kv_engine 一致（文件头、ETCD 元数据、slot_mapping、payload_meta）。
    """

    def __init__(self, vllm_config: Any, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        # 复用 vLLM 的 logger 配置，确保日志可见
        try:
            from vllm.logger import init_logger as _init_logger  # type: ignore
            self._logger = _init_logger(self.__class__.__name__)
        except Exception:  # pragma: no cover
            self._logger = logging.getLogger(self.__class__.__name__)
        # 初始化存储与元数据，仅复用 storage/etcd 模块（不再依赖旧 engine）
        try:
            cfg_path = getattr(getattr(vllm_config, 'kv_transfer_config', None), 'config_path', None)
        except Exception:
            cfg_path = None
        # 合并 config.json，确保 kv_transfer_config 生效
        try:
            base_cfg = load_config_from_json(cfg_path) if cfg_path is not None else load_config_from_json()
        except Exception:
            base_cfg = None
        if base_cfg is None:
            base_cfg = SimpleNamespace(kv_transfer_config=getattr(vllm_config, 'kv_transfer_config', SimpleNamespace()))
        else:
            # 覆盖 rank/local_rank/engine_id 为 vllm_config 中的值（若存在）
            if hasattr(vllm_config, 'rank'):
                base_cfg.rank = getattr(vllm_config, 'rank')
            if hasattr(vllm_config, 'local_rank'):
                base_cfg.local_rank = getattr(vllm_config, 'local_rank')
            if hasattr(vllm_config, 'engine_id'):
                base_cfg.engine_id = getattr(vllm_config, 'engine_id')
        # 构建存储实例（v1 包装器）
        self._v1_storage: V1Storage = create_v1_storage(base_cfg)
        # etcd 元数据与缓存（v1 包装器）
        endpoints = getattr(getattr(base_cfg, 'kv_transfer_config', SimpleNamespace()), 'etcd_endpoints', ["127.0.0.1:2379"])  # type: ignore
        self._meta_manager = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")
        self._v1_meta = V1MetadataClient(manager=self._meta_manager)
        # 运行时状态
        self._kv_caches: Dict[str, torch.Tensor] = {}
        self._last_forward_context: Any = None
        # 每个请求已保存到的 token 边界（已持久化的 token 数）
        self._req_last_stored: Dict[str, int] = {}
        # scheduler 侧跟踪 load 规格
        self._load_specs: Dict[str, _ReqLoadSpec] = {}
        # request trackers (仅保存 token_ids 已调度部分) 简化实现
        self._request_tokens: Dict[str, List[int]] = {}
        self._unfinished_requests: Dict[str, Any] = {}
        # 记录每个请求的已分配 block ids（用于构造 slot_mapping）
        self._req_block_ids = {}
        # 记录每个请求最近一次构建的 slot_mapping（1D 索引）
        self._req_slot_mapping = {}
        # 记录调度器本步为每个请求分配/调度的 token 数（来自 update_state_after_alloc 的 num_external_tokens）
        self._req_sched_tokens: Dict[str, int] = {}
        # 逐层收集：layer 顺序与名称映射
        self._kv_layer_names: List[str] = []
        self._layer_name_to_idx: Dict[str, int] = {}
        # 逐层暂存的待写切片：key=(req_id, start, end) -> {layer_idx: {k,v,slot,tokens}}
        self._pending_store: Dict[Tuple[str, int, int], Dict[int, Dict[str, torch.Tensor]]] = {}
        # req -> session_id（字符串）
        self._req_session_id: Dict[str, str] = {}
        # 保存当前步骤构造的元数据供 worker 使用
        self._current_metadata = None
        try:
            self._chunk_size = int(getattr(getattr(vllm_config, 'kv_transfer_config', None), 'chunk_size', 256) or 256)
        except Exception:
            self._chunk_size = 256
        # 跳过最后 n tokens 保存（可选）
        try:
            self._skip_last_n = int(getattr(getattr(vllm_config, 'kv_transfer_config', None), 'skip_last_n_tokens', 0) or 0)
        except Exception:
            self._skip_last_n = 0
        # 日志关键配置
        try:
            st_dir = getattr(base_cfg.kv_transfer_config, 'storage_dir', None)  # type: ignore
        except Exception:
            st_dir = None
        eps = endpoints
        self._logger.info("[v1_engine] initialized storage_dir=%s etcd_endpoints=%s (synchronous load/save)", st_dir, eps)

    # ---------------- Worker-side ----------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches or {}
        try:
            # 记录层顺序，便于 layer_name -> idx 映射
            self._kv_layer_names = list(self._kv_caches.keys())
            self._layer_name_to_idx = {name: i for i, name in enumerate(self._kv_layer_names)}
        except Exception:
            self._kv_layer_names = []
            self._layer_name_to_idx = {}
        self._logger.info("[v1_engine] registered kv layers=%d", len(self._kv_caches))

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        # 持有 forward_context 供后续 wait_for_save 使用
        self._last_forward_context = forward_context
        meta = self._safe_get_metadata()
        if meta is None or not meta.requests:
            return
        self._logger.debug("[v1_engine] start_load_kv: meta.requests=%d", len(meta.requests))
        # 若判定可加载，则一次性按完整 prompt 长度加载
        for plan in meta.requests:
            try:
                if plan.load_spans:
                    for span in plan.load_spans:
                        self._load_by_plan_slice(forward_context, plan, span.start, span.end)
                else:
                    # 无明确切片则按整个 token_ids
                    self._load_by_plan_slice(forward_context, plan, 0, len(plan.token_ids))
            except Exception:
                self._logger.exception("start_load_kv: failed for req=%s", getattr(plan, 'req_id', '?'))

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs) -> None:
        """逐层收集：从当前层的 paged KV 中提取本轮需要持久化的切片，暂存于内存。

        等到 wait_for_save 再按聚合（或逐层）方式真正写入存储。
        """
        try:
            meta = self._safe_get_metadata()
            if meta is None or not meta.requests:
                return
            # 确定 layer_idx
            try:
                layer_idx = int(self._layer_name_to_idx.get(layer_name, -1))
                if layer_idx < 0:
                    # 若未注册，尝试动态插入
                    nid = len(self._layer_name_to_idx)
                    self._layer_name_to_idx[layer_name] = nid
                    layer_idx = nid
            except Exception:
                layer_idx = 0

            # 拆分 K/V 源
            def _split_layer(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if t.dim() < 2 or t.size(0) != 2:
                    raise ValueError(f"无法识别的kv_layer形状: {tuple(t.shape)}")
                return t[0], t[1]

            k_src, v_src = _split_layer(kv_layer)

            for plan in meta.requests:
                req_id = plan.req_id
                # 记录 req -> session 映射
                self._req_session_id[req_id] = plan.session_id
                if not plan.store_spans:
                    continue
                # 计划内每个切片都做一次提取
                for span in plan.store_spans:
                    s = int(getattr(span, 'start', 0))
                    e = int(getattr(span, 'end', len(plan.token_ids)))
                    if e <= s:
                        continue
                    # 目标槽位：使用计划内的 slot_mapping
                    try:
                        slot_list = plan.slot_mapping[s:e]
                        slot = torch.tensor(slot_list, dtype=torch.long, device=kv_layer.device)
                    except Exception:
                        slot = torch.arange(e - s, dtype=torch.long, device=kv_layer.device)

                    # 从当前层的 paged KV 选择对应切片
                    def _take(src: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
                        if src.dim() == 3:
                            return src.index_select(0, slots)
                        elif src.dim() == 4:
                            # 假定 batch 维度在 0
                            take = min(int(src.shape[1]), int(slots.numel()))
                            if take <= 0:
                                return src.new_empty((0,))
                            return src[0, :take]
                        else:
                            return src

                    k_slice = _take(k_src, slot)
                    v_slice = _take(v_src, slot)
                    if k_slice.numel() == 0 or v_slice.numel() == 0:
                        continue

                    # 转存 CPU，减少 GPU 压力
                    try:
                        k_slice = k_slice.contiguous().cpu()
                        v_slice = v_slice.contiguous().cpu()
                        slot_cpu = slot.contiguous().cpu()
                        tokens = torch.tensor(plan.token_ids[s:e], dtype=torch.long)
                    except Exception:
                        # 回退保持在当前设备
                        slot_cpu = slot
                        tokens = torch.tensor(plan.token_ids[s:e], dtype=torch.long, device=slot.device)

                    key = (req_id, s, e)
                    bucket = self._pending_store.setdefault(key, {})
                    bucket[layer_idx] = {
                        'k': k_slice,
                        'v': v_slice,
                        'slot': slot_cpu,
                        'tokens': tokens,
                    }
        except Exception:
            self._logger.exception("save_kv_layer failed for layer=%s", layer_name)
        return

    def wait_for_save(self) -> None:
        try:
            fc = self._last_forward_context
            if fc is None:
                return
            meta = self._safe_get_metadata()
            if meta is None or not meta.requests:
                return
            for plan in meta.requests:
                try:
                    if not plan.store_spans:
                        continue
                    for span in plan.store_spans:
                        # 优先刷写逐层收集的 pending；不足时回退从 paged KV 采集
                        if not self._flush_pending_span(fc, plan, span):
                            self._save_by_plan_slice(fc, plan, span.start, span.end)
                except Exception:
                    self._logger.exception("wait_for_save: store failed for req=%s", getattr(plan, 'req_id', '?'))
            # 可选：同步等待写入完成，方便立即看到落盘
            try:
                vcfg = getattr(self.vllm_config, "kv_transfer_config", None)
                force_sync = bool(getattr(vcfg, "force_sync_store", False))
            except Exception:
                force_sync = False
            if force_sync:
                futures = list(getattr(self._engine, "_futures", []) or [])
                self._logger.debug("[v1_engine] force_sync_store waiting %d futures", len(futures))
                for f in futures:
                    try:
                        f.result(timeout=10)
                    except Exception:
                        pass
        except Exception:
            self._logger.exception("wait_for_save unexpected failure")

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # 同步实现且不占用异步队列：不要回传任何已完成ID
        # 否则调度器会尝试对未登记的请求做二次释放，触发 KeyError。
        return set(), set()

    # ---------------- Scheduler-side ----------------
    def get_num_new_matched_tokens(self, request: "Any", num_computed_tokens: int) -> tuple[int, bool]:
        """调度侧：基于 etcd 元数据判断是否命中并可加载。

        简化策略：仅支持完整 prompt 命中；部分命中返回 0。
        """
        try:
            tokens = list(getattr(request, 'prompt_token_ids', []))
            prompt_len = int(len(tokens))
            if prompt_len == 0:
                return 0, False
            sess_bytes = self._resolve_session_bytes(getattr(request, 'request_id', 'v1_session'))
            file_path = self._make_key(torch.tensor(tokens, dtype=torch.long), sess_bytes, 0)
            meta = self._v1_meta.get_metadata(key=file_path, layer_id=0, session_id=sess_bytes)
            if meta is None or getattr(meta, 'status', 0) != 1:
                self._load_specs[getattr(request, 'request_id', getattr(request, 'req_id', ''))] = _ReqLoadSpec(num_computed_tokens, 0, False)
                return 0, False
            # 命中：允许加载，need_allocate = cached - computed - 1
            need_allocate = max(0, prompt_len - num_computed_tokens - 1)
            self._load_specs[getattr(request, 'request_id', getattr(request, 'req_id', ''))] = _ReqLoadSpec(num_computed_tokens, prompt_len, need_allocate > 0)
            self._logger.info("[v1_engine] (etcd) match req=%s cached=%d computed=%d need=%d", getattr(request, 'request_id', '?'), prompt_len, num_computed_tokens, need_allocate)
            return need_allocate, False
        except Exception:
            return 0, False

    def update_state_after_alloc(self, request: "Any", blocks: "Any", num_external_tokens: int):
        # 记录 unfinished 请求供后续 build_connector_meta/start_load_kv 使用
        req_id = getattr(request, 'request_id', getattr(request, 'req_id', ''))
        self._unfinished_requests[req_id] = request
        # 记录本步调度的 token 数，供判定是否为最后一次 prefill
        try:
            self._req_sched_tokens[req_id] = int(num_external_tokens or 0)
        except Exception:
            self._req_sched_tokens[req_id] = 0
        # 维护 token id 序列（prompt token + 已调度新 token）简化：直接使用 prompt_token_ids
        toks = list(getattr(request, 'prompt_token_ids', []))
        self._request_tokens[req_id] = toks
        # 记录/合并本步分配的 block ids（用于后续 slot_mapping 推导）
        try:
            new_block_ids: List[int] = []
            if hasattr(blocks, 'get_block_ids') and callable(getattr(blocks, 'get_block_ids')):
                bid = blocks.get_block_ids()
                # 兼容多组结构，取第一组
                if isinstance(bid, (list, tuple)) and len(bid) > 0:
                    new_block_ids = bid[0] if isinstance(bid[0], list) else list(bid)
            elif hasattr(blocks, 'block_ids'):
                bid = getattr(blocks, 'block_ids')
                if isinstance(bid, (list, tuple)) and len(bid) > 0:
                    new_block_ids = bid[0] if isinstance(bid[0], list) else list(bid)
            if new_block_ids:
                prev = self._req_block_ids.get(req_id, [])
                # 简单合并去重，保持顺序：
                merged: List[int] = list(prev)
                for x in new_block_ids:
                    if x not in merged:
                        merged.append(int(x))
                self._req_block_ids[req_id] = merged
        except Exception:
            pass
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        # 构造元数据：针对新请求与继续调度的请求
        meta = _V1Metadata()
        # finished 请求清理
        finished: Set[str] = set(getattr(scheduler_output, 'finished_req_ids', []) or [])
        for fid in finished:
            self._unfinished_requests.pop(fid, None)
            self._request_tokens.pop(fid, None)
            self._req_last_stored.pop(fid, None)

        # 新请求（含首次调度）
        new_reqs = getattr(scheduler_output, 'scheduled_new_reqs', []) or []
        for req in new_reqs:
            req_id = getattr(req, 'request_id', getattr(req, 'req_id', ''))
            toks = list(getattr(req, 'prompt_token_ids', []))
            num_comp = int(getattr(req, 'num_computed_tokens', 0) or 0)
            load_spec = self._load_specs.pop(req_id, None)
            # 是否最后一次 prefill: prompt 全部被调度
            # 优先使用 update_state_after_alloc 提供的 num_external_tokens；
            # 若不可用，再回退到 scheduler_output 提供的统计；最后默认为 0。
            num_sched = int(self._req_sched_tokens.get(req_id, 0))
            if num_sched == 0:
                try:
                    num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                except Exception:
                    num_sched = 0
            is_last_prefill = (num_comp + num_sched) >= len(toks)
            # 保存策略：初始 skip_leading 为已持久化 token 数（上次为 0）
            plan = self._build_request_plan(req_id, toks, load_spec, is_last_prefill, num_sched_tokens=num_sched)
            meta.add_request(plan)
            # 保存 tracker
            self._request_tokens[req_id] = toks
            self._unfinished_requests[req_id] = req
            self._req_slot_mapping[req_id] = torch.tensor(plan.slot_mapping, dtype=torch.long)
            # 本轮统计已消费，避免下一轮误用旧值
            self._req_sched_tokens.pop(req_id, None)

        # 已在缓存中继续调度的请求（cached_reqs）
        cached_reqs = getattr(scheduler_output, 'scheduled_cached_reqs', None)
        # 兼容不同 vLLM 版本：list 或对象
        if isinstance(cached_reqs, list):
            iterable = cached_reqs
            for req in iterable:
                req_id = getattr(req, 'request_id', getattr(req, 'req_id', ''))
                toks = self._request_tokens.get(req_id, list(getattr(req, 'prompt_token_ids', [])))
                num_comp = int(getattr(req, 'num_computed_tokens', 0) or 0)
                load_spec = None  # 对继续的请求不再尝试 load
                num_sched = int(self._req_sched_tokens.get(req_id, 0) or 0)
                if num_sched == 0:
                    try:
                        num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                    except Exception:
                        num_sched = 0
                is_last_prefill = (num_comp + num_sched) >= len(toks)
                plan = self._build_request_plan(req_id, toks, load_spec, is_last_prefill, num_sched_tokens=num_sched)
                meta.add_request(plan)
                self._request_tokens[req_id] = toks
                self._req_slot_mapping[req_id] = torch.tensor(plan.slot_mapping, dtype=torch.long)
                self._req_sched_tokens.pop(req_id, None)
        else:
            try:
                for i, req_id in enumerate(getattr(cached_reqs, 'req_ids', []) or []):
                    req = self._unfinished_requests.get(req_id)
                    toks = self._request_tokens.get(req_id, list(getattr(req, 'prompt_token_ids', [])))
                    num_comp = int(getattr(req, 'num_computed_tokens', 0) or 0)
                    load_spec = None
                    num_sched = int(self._req_sched_tokens.get(req_id, 0) or 0)
                    if num_sched == 0:
                        try:
                            num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                        except Exception:
                            num_sched = 0
                    is_last_prefill = (num_comp + num_sched) >= len(toks)
                    plan = self._build_request_plan(req_id, toks, load_spec, is_last_prefill, num_sched_tokens=num_sched)
                    meta.add_request(plan)
                    self._request_tokens[req_id] = toks
                    self._req_slot_mapping[req_id] = torch.tensor(plan.slot_mapping, dtype=torch.long)
                    self._req_sched_tokens.pop(req_id, None)
            except Exception:
                pass

        # 保存 forward_context
        try:
            self._last_forward_context = getattr(scheduler_output, 'forward_context', None)
        except Exception:
            self._last_forward_context = None
        self._current_metadata = meta
        return meta

    def request_finished(self, request: "Any", block_ids: List[int]) -> tuple[bool, Optional[dict[str, Any]]]:
        # 不阻塞释放
        self._req_last_stored.pop(getattr(request, "request_id", ""), None)
        return False, None

    def take_events(self):
        return []

    def update_connector_output(self, connector_output: Any):
        # 当前无状态需要更新
        return

    # ---------------- Internal helpers ----------------
    def _safe_get_metadata(self) -> Optional[_V1Metadata]:
        try:
            meta = self._get_connector_metadata()
        except Exception:
            meta = self._current_metadata
        if isinstance(meta, _V1Metadata):
            return meta
        if isinstance(meta, KVConnectorMetadata) and hasattr(meta, "requests"):
            # vLLM 可能反序列化为普通对象；尽量构造 _V1Metadata 视图
            new_meta = _V1Metadata()
            new_meta.requests = getattr(meta, "requests")
            return new_meta
        return None

    def _retrieve_full(self, fc: Any, req: Any) -> None:
        mi = self._build_model_input(fc, req, None)
        if mi is None:
            return
        rs = should_retrieve(mi)
        kvs = self._collect_kv(fc, None)
        if not kvs:
            self._logger.warning("[v1_engine] retrieve: no kv_caches available (registered=%d fc_has=%s)",
                                 len(self._kv_caches), hasattr(fc, "kv_caches"))
        model_exec = self._get_model_exec(fc)
        if model_exec is None:
            self._logger.warning("[v1_engine] retrieve: forward_context has no model; skip")
            return
        retrieve_kv(model_exec, mi, kvs, rs)
        self._logger.debug("[retrieve] req=%s tokens=%d", getattr(req, "request_id", "?"), mi.input_tokens.shape[0])

    def _load_by_plan_slice(self, fc: Any, plan: _RequestPlan, start: int, end: int) -> None:
        # 读取 payload 并注入到 paged KV 缓存
        try:
            if end <= start:
                return
            sess = plan.session_id.encode('utf-8', errors='ignore')
            tokens = torch.tensor(plan.token_ids[start:end], dtype=torch.long)
            file_path = self._make_key(tokens, sess, 0)
            kv_bytes = self._v1_storage.download(file_path)
            if kv_bytes is None:
                self._logger.debug("[v1_engine] load miss: %s", file_path)
                return
            info = self._v1_storage.extract_payload_info(kv_bytes)
            k_tensor, v_tensor = self._v1_storage.unpack_kv_data(kv_bytes)
            if k_tensor is None or v_tensor is None:
                self._logger.warning("[v1_engine] unpack_kv_data failed: %s", file_path)
                return
            # 构造 slot mapping（优先用计划的）
            sm = torch.tensor(plan.slot_mapping[start:end], dtype=torch.long) if plan.slot_mapping else torch.arange(end-start, dtype=torch.long)
            # 注入
            self._inject_into_caches(fc, k_tensor, v_tensor, sm)
            self._logger.debug("[v1_engine] loaded req=%s slice=[%d,%d) from %s", plan.req_id, start, end, file_path)
        except Exception:
            self._logger.exception("_load_by_plan_slice failed for req=%s", plan.req_id)

    def _store_slice(self, fc: Any, req: Any, start: int, end: int) -> None:
        if end <= start:
            return
        mi = self._build_model_input(fc, req, (start, end))
        if mi is None:
            return
        ss = should_store(mi)
        kvs = self._collect_kv(fc, (start, end))
        if not kvs:
            self._logger.warning("[v1_engine] store: no kv_caches to persist for req=%s span=[%d,%d)", getattr(req, "request_id", "?"), start, end)
        model_exec = self._get_model_exec(fc)
        if model_exec is None:
            self._logger.warning("[v1_engine] store: forward_context has no model; skip span=[%d,%d)", start, end)
            return
        store_kv(self.vllm_config.model_config, self.vllm_config.parallel_config, None, model_exec, mi, kvs, ss, None)
        self._logger.debug("[store] req=%s span=[%d,%d) len=%d", getattr(req, "request_id", "?"), start, end, end-start)

    def _save_by_plan_slice(self, fc: Any, plan: _RequestPlan, start: int, end: int) -> None:
        if end <= start:
            return
        try:
            sess = plan.session_id.encode('utf-8', errors='ignore')
            tokens = torch.tensor(plan.token_ids[start:end], dtype=torch.long)
            slot = torch.tensor(plan.slot_mapping[start:end], dtype=torch.long) if plan.slot_mapping else torch.arange(end-start, dtype=torch.long)
            # 从 paged KV 缓存收集 KV，按 slot 顺序重排成 [num_layers, seq_len, ...]
            k_all, v_all = self._gather_from_caches(fc, slot)
            if k_all is None or v_all is None or k_all.numel() == 0:
                self._logger.warning("[v1_engine] no kv to store for req=%s slice=[%d,%d)", plan.req_id, start, end)
                return
            payload_meta = {
                'schema_version': 1,
                'tokens_hash': self._tensor_hash(tokens),
                'num_layers': int(k_all.shape[0]) if k_all.dim() >= 2 else 0,
                'kv_dtype': str(k_all.dtype),
                'kv_tail_shape': list(k_all.shape[2:]) if k_all.dim() >= 3 else [],
                'slots_len': int(slot.numel()),
                'token_offset': int(start),
                'block_size': int(end - start),
            }
            data = self._v1_storage.pack_full_payload(k_all, v_all, tokens, torch.ones_like(tokens, dtype=torch.bool), slot, payload_meta)
            file_path = self._make_key(tokens, sess, 0)
            ok = self._v1_storage.upload(file_path, data)
            if not ok:
                self._logger.warning("[v1_engine] upload failed: %s", file_path)
                return
            # 写 etcd 元数据
            expire_time = int(getattr(getattr(self.vllm_config, 'kv_transfer_config', SimpleNamespace()), 'kv_expire_time', 86400) or 86400)
            meta = KVMetadata(
                session_id=sess[:16].ljust(16, b"\x00"),
                layer_id=0,
                token_idx=f"{start}-{end}",
                file_path=file_path,
                file_size=len(data),
                create_time=int(time.time()),
                last_access=int(time.time()),
                expire_time=expire_time,
                replica_locations=[b"" for _ in range(3)],
                status=1,
                schema_version=1,
                ext_flags=0,
                ext_data=b"",
                ext_data_len=0,
            )
            self._v1_meta.put_metadata(meta)
            self._logger.debug("[v1_engine] stored req=%s slice=[%d,%d) -> %s", plan.req_id, start, end, file_path)
        except Exception:
            self._logger.exception("_save_by_plan_slice failed for req=%s", plan.req_id)

    def _build_model_input(self, fc: Any, req: Any, span: Optional[Tuple[int, int]]) -> Optional[SimpleNamespace]:
        inp = getattr(req, "input_ids", None)
        if inp is None:
            # vLLM 请求对象常用 prompt_token_ids 表示完整 prompt
            inp = getattr(req, "prompt_token_ids", None)
        if inp is None:
            return None
        dev = self._infer_device(fc)
        tokens = torch.tensor(inp, dtype=torch.long, device=dev) if not torch.is_tensor(inp) else inp.to(dev)
        if span is not None:
            s, e = span
            tokens = tokens[s:e]
        mi = SimpleNamespace()
        mi.input_tokens = tokens
        am = SimpleNamespace()
        am.seq_lens = [int(tokens.shape[0])]
        # 优先使用调度侧推导的 slot_mapping；其次尝试从 forward_context.attn_metadata.slot_mapping 获取；最后回退 arange
        try:
            req_id = getattr(req, 'request_id', getattr(req, 'req_id', ''))
        except Exception:
            req_id = ''
        sm: Optional[torch.Tensor] = None
        try:
            full_sm = self._req_slot_mapping.get(req_id)
            if full_sm is not None and torch.is_tensor(full_sm):
                total_len = int(getattr(req, 'prompt_len', len(getattr(req, 'prompt_token_ids', []))) or tokens.shape[0])
                # full_sm 长度通常等于 prompt_len；需要根据 span 对应切片
                if span is None:
                    sm = full_sm[:tokens.shape[0]].to(device=dev)
                else:
                    s, e = span
                    sm = full_sm[s:e].to(device=dev)
        except Exception:
            sm = None
        if sm is None:
            try:
                fcam = getattr(fc, 'attn_metadata', None)
                if fcam is not None:
                    orig_sm = getattr(fcam, 'slot_mapping', None)
                    if torch.is_tensor(orig_sm):
                        if orig_sm.dim() == 1:
                            # 无法定位到该 request 的 start_pos，这里回退使用前 tokens.shape[0] 长度
                            sm = orig_sm[:tokens.shape[0]].to(device=dev)
                        elif orig_sm.dim() >= 2:
                            # 尝试取第0序列
                            sm = orig_sm[0][:tokens.shape[0]].to(device=dev)
            except Exception:
                sm = None
        if sm is None:
            sm = torch.arange(tokens.shape[0], device=dev)
        am.slot_mapping = sm
        mi.attn_metadata = am
        mi.session_id = self._resolve_session_bytes(getattr(req, "request_id", "v1_session"))
        mi.layer_id = 0
        if span is not None:
            mi.payload_meta = {"token_offset": int(span[0]), "block_size": int(tokens.shape[0])}
        return mi

    def _build_request_plan(self, req_id: str, tokens: List[int],
                            load_spec: Optional[_ReqLoadSpec], is_last_prefill: bool,
                            num_sched_tokens: Optional[int] = None) -> _RequestPlan:
        slot_tensor = self._build_slot_mapping_from_blocks(req_id, len(tokens))
        slot_list = (
            slot_tensor.tolist()
            if torch.is_tensor(slot_tensor)
            else [int(x) for x in slot_tensor]
        )
        plan = _RequestPlan(
            req_id=req_id,
            session_id=self._resolve_session_id(req_id),
            token_ids=[int(t) for t in tokens],
            slot_mapping=[int(s) for s in slot_list],
            is_last_prefill=is_last_prefill,
        )
        if load_spec and load_spec.can_load:
            start = int(load_spec.vllm_cached_tokens)
            end = int(load_spec.cached_tokens)
            if end > start:
                plan.load_spans.append(_LoadSpan(start=start, end=end))
        prev = int(self._req_last_stored.get(req_id, 0))
        # 生成存储计划：
        # - 若为最后一次 prefill，则存储到完整 prompt 长度。
        # - 否则尝试增量存储本步新调度的 token（如果知道 num_sched_tokens）。
        if is_last_prefill and len(tokens) > prev:
            plan.store_spans.append(_StoreSpan(start=prev, end=len(tokens)))
            self._req_last_stored[req_id] = len(tokens)
        else:
            try:
                ns = int(num_sched_tokens or 0)
            except Exception:
                ns = 0
            if ns > 0 and len(tokens) > prev:
                end = min(len(tokens), prev + ns)
                if end > prev:
                    plan.store_spans.append(_StoreSpan(start=prev, end=end))
                    self._req_last_stored[req_id] = end
        return plan

    def _resolve_session_id(self, req_id: str) -> str:
        try:
            sid_src = getattr(getattr(self.vllm_config, "kv_transfer_config", None), "engine_id", None)
            if sid_src:
                return str(sid_src)
        except Exception:
            pass
        return req_id or "v1_session"

    def _resolve_session_bytes(self, req_id: str) -> bytes:
        try:
            sid_src = getattr(getattr(self.vllm_config, "kv_transfer_config", None), "engine_id", None)
            if sid_src:
                return str(sid_src).encode('utf-8')
        except Exception:
            pass
        return (req_id or "v1_session").encode('utf-8')

    def _build_model_input_from_plan(self, plan: _RequestPlan, span: Any) -> Optional[SimpleNamespace]:
        start = int(getattr(span, "start", 0))
        end = int(getattr(span, "end", len(plan.token_ids)))
        if end <= start:
            return None
        token_slice = plan.token_ids[start:end]
        if not token_slice:
            return None
        slot_source = plan.slot_mapping if plan.slot_mapping else list(range(len(plan.token_ids)))
        slot_slice = slot_source[start:end]
        fc = self._last_forward_context
        dev = self._infer_device(fc)
        tokens = torch.tensor(token_slice, dtype=torch.long, device=dev)
        slot_tensor = torch.tensor(slot_slice[: len(token_slice)], dtype=torch.long, device=dev)
        mi = SimpleNamespace()
        mi.input_tokens = tokens
        am = SimpleNamespace()
        am.seq_lens = [int(tokens.shape[0])]
        am.slot_mapping = slot_tensor
        mi.attn_metadata = am
        sid = plan.session_id.encode("utf-8", errors="ignore")
        mi.session_id = sid or b"v1_session"
        mi.layer_id = 0
        mi.payload_meta = {"token_offset": start, "block_size": int(tokens.shape[0])}
        return mi

    def _build_slot_mapping_from_blocks(self, req_id: str, token_len: int) -> torch.Tensor:
        """根据记录的 block_ids 和 vLLM 的 block_size 构造 1D slot_mapping。

        若无可用 block_ids，则回退为 arange。
        """
        try:
            block_ids = self._req_block_ids.get(req_id, [])
            if not block_ids:
                return torch.arange(token_len, dtype=torch.long)
            # 获取 block_size（优先 vLLM v1 的 cache_config.block_size，其次回退旧字段）
            bs = None
            try:
                cc = getattr(self.vllm_config, 'cache_config', None)
                if cc is not None:
                    bs = int(getattr(cc, 'block_size', 0) or 0)
            except Exception:
                bs = None
            if not bs:
                try:
                    bs = int(getattr(getattr(self.vllm_config, 'v1_config', None), 'gpu_block_size', 0) or 0)
                except Exception:
                    bs = None
            if not bs:
                try:
                    bs = int(getattr(self.vllm_config, 'gpu_block_size', 0) or 0)
                except Exception:
                    bs = None
            if not bs:
                # 保底：使用 1，等价于直接展开
                return torch.arange(token_len, dtype=torch.long)
            block_offsets = torch.arange(0, bs, dtype=torch.long)
            bids = torch.tensor(block_ids, dtype=torch.long).reshape((-1, 1))
            sm = (bids * bs + block_offsets.reshape((1, bs))).flatten()[:token_len]
            return sm
        except Exception:
            return torch.arange(token_len, dtype=torch.long)

    # ---------------- Low-level IO helpers (no legacy engine deps) ----------------
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        import hashlib
        if tensor.numel() == 0:
            return "empty"
        tensor_bytes = tensor.cpu().numpy().tobytes()
        return hashlib.blake2b(tensor_bytes, digest_size=24).hexdigest()

    def _make_key(self, input_tokens: torch.Tensor, session_id: Optional[bytes] = None, layer_id: Optional[int] = None) -> str:
        seq_hash = self._tensor_hash(input_tokens)
        if session_id is None:
            session_id = b"session_0000"
        if layer_id is None:
            layer_id = 0
        session_str = session_id.decode('utf-8', errors='ignore') if isinstance(session_id, (bytes, bytearray)) else str(session_id)
        return f"kv_{session_str}_layer_{layer_id}_{seq_hash}.pt"

    def _inject_into_caches(self, fc: Any, k_all: torch.Tensor, v_all: torch.Tensor, slot: torch.Tensor) -> None:
        try:
            caches = self._kv_caches if self._kv_caches else getattr(fc, 'kv_caches', {})
            kvs = list(caches.values())
            num_layers = len(kvs)
            if num_layers == 0:
                return
            for layer_idx in range(num_layers):
                kv_cache = kvs[layer_idx]
                key_cache = kv_cache[0]
                value_cache = kv_cache[1]
                k_src = k_all[layer_idx]
                v_src = v_all[layer_idx]
                if key_cache.dim() == 3:
                    limit = min(int(k_src.shape[0]), int(slot.numel()))
                    if limit <= 0:
                        continue
                    key_cache[slot[:limit]] = k_src[:limit].to(key_cache.dtype)
                    value_cache[slot[:limit]] = v_src[:limit].to(value_cache.dtype)
                elif key_cache.dim() == 4:
                    write_len = min(int(k_src.shape[0]), int(key_cache.shape[1]))
                    if write_len <= 0:
                        continue
                    key_cache[0, :write_len] = k_src[:write_len].to(key_cache.dtype)
                    value_cache[0, :write_len] = v_src[:write_len].to(value_cache.dtype)
        except Exception:
            self._logger.exception("_inject_into_caches failed")

    def _gather_from_caches(self, fc: Any, slot: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            caches = self._kv_caches if self._kv_caches else getattr(fc, 'kv_caches', {})
            kvs = list(caches.values())
            if not kvs:
                return None, None
            keys, values = [], []
            for kv_cache in kvs:
                key_cache = kv_cache[0]
                value_cache = kv_cache[1]
                if key_cache.dim() == 3:
                    k = key_cache[slot]
                    v = value_cache[slot]
                elif key_cache.dim() == 4:
                    take = min(int(key_cache.shape[1]), int(slot.numel()))
                    k = key_cache[0, :take]
                    v = value_cache[0, :take]
                else:
                    continue
                keys.append(k.unsqueeze(0))
                values.append(v.unsqueeze(0))
            if not keys or not values:
                return None, None
            return torch.cat(keys, dim=0), torch.cat(values, dim=0)
        except Exception:
            self._logger.exception("_gather_from_caches failed")
            return None, None

    def _collect_kv(self, fc: Any, span: Optional[Tuple[int, int]]) -> List[torch.Tensor]:
        try:
            caches = self._kv_caches if self._kv_caches else getattr(fc, "kv_caches", {})
            tensors = list(caches.values())
            if not tensors:
                return []
            if span is None:
                return tensors
            s, e = span
            sliced: List[torch.Tensor] = []
            for t in tensors:
                try:
                    k, v = t[0], t[1]
                    if k.dim() == 4:
                        sliced.append(torch.stack([k[0:1, s:e].contiguous(), v[0:1, s:e].contiguous()], dim=0))
                    elif k.dim() == 3:
                        sliced.append(torch.stack([k[s:e].contiguous(), v[s:e].contiguous()], dim=0))
                    else:
                        sliced.append(t)
                except Exception:
                    sliced.append(t)
            return sliced
        except Exception:
            return []

    def _infer_device(self, fc: Any) -> torch.device:
        # Prefer model param device
        try:
            me = self._get_model_exec(fc)
            if me is not None:
                return next(me.parameters()).device
        except Exception:
            pass
        # Fallback: try from registered kv cache tensors
        try:
            if self._kv_caches:
                t = next(iter(self._kv_caches.values()))
                return t.device
        except Exception:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_model_exec(self, fc: Any) -> Optional[Any]:
        """Best-effort extraction of model_executable from forward_context.

        Tries common attribute names across vLLM variants and our tests.
        """
        try:
            cand = getattr(fc, "model", None)
            if cand is not None:
                return cand
        except Exception:
            pass
        try:
            cand = getattr(fc, "model_executable", None)
            if cand is not None:
                return cand
        except Exception:
            pass
        try:
            eng = getattr(fc, "engine", None)
            cand = getattr(eng, "model", None)
            if cand is not None:
                return cand
        except Exception:
            pass
        # last resort: some configs may stash it in vllm_config
        try:
            cand = getattr(self.vllm_config, "model_executable", None)
            if cand is not None:
                return cand
        except Exception:
            pass
        return None


# ---------------- Module-level helpers for connector thin forwarding ----------------
_CORE_SINGLETON: Optional[V1KVEngineImpl] = None


def init_v1_engine(vllm_config: Any, role: Any = None) -> V1KVEngineImpl:
    global _CORE_SINGLETON
    if _CORE_SINGLETON is None:
        # role 仅用于日志/调试，不强制类型
        real_role = role if role is not None else KVConnectorRole.WORKER
        _CORE_SINGLETON = V1KVEngineImpl(vllm_config, real_role)
    return _CORE_SINGLETON


def destroy_v1_engine():
    global _CORE_SINGLETON
    try:
        # stop metadata cache async thread if any
        try:
            if _CORE_SINGLETON is not None and hasattr(_CORE_SINGLETON, '_v1_meta'):
                getattr(_CORE_SINGLETON._v1_meta, 'stop', lambda: None)()
        except Exception:
            pass
    finally:
        _CORE_SINGLETON = None


def v1_should_store(model_input: Any) -> Any:
    # deprecated in lightweight engine
    return None


def v1_store_kv(model_config: Any, parallel_config: Any, sampler: Any, model_executable: Any,
                model_input: Any, kv_caches: List[torch.Tensor], store_status: Any,
                hidden_states: Optional[torch.Tensor]) -> None:
    return None


def v1_should_retrieve(model_input: Any) -> Any:
    return None


def v1_retrieve_kv(model_executable: Any, model_input: Any,
                   kv_caches: List[torch.Tensor], retrieve_status: Any) -> Any:
    return None
    def _flush_pending_span(self, fc: Any, plan: _RequestPlan, span: _StoreSpan) -> bool:
        """Flush pending per-layer slices if available.

        Returns True if flushed successfully (layered: any flushed; aggregate: all layers present),
        False to request fallback to gather-from-cache path.
        """
        try:
            start = int(getattr(span, 'start', 0))
            end = int(getattr(span, 'end', len(plan.token_ids)))
            if end <= start:
                return True
            key = (plan.req_id, start, end)
            bucket = self._pending_store.get(key)
            if not bucket:
                return False
            sess = self._resolve_session_bytes(plan.req_id)
            tokens = torch.tensor(plan.token_ids[start:end], dtype=torch.long)
            slot = torch.tensor(plan.slot_mapping[start:end], dtype=torch.long)
            if self._storage_mode == 'layered' and self._v1_layered is not None:
                # 逐层写：有多少层刷多少层
                flushed_any = False
                for lid, item in list(bucket.items()):
                    try:
                        k_slice = item['k']
                        v_slice = item['v']
                        slot_slice = item['slot']
                        self._v1_layered.upload_layer_slice(sess, int(lid), tokens, k_slice, v_slice, slot_slice, start, end, payload_meta_extra=None)
                        # 写 etcd
                        expire_time = int(getattr(getattr(self.vllm_config, 'kv_transfer_config', SimpleNamespace()), 'kv_expire_time', 86400) or 86400)
                        file_path = self._v1_layered._file_name(sess, int(lid), tokens, start, end)
                        meta = KVMetadata(
                            session_id=sess[:16].ljust(16, b"\x00"),
                            layer_id=int(lid),
                            token_idx=f"{start}-{end}",
                            file_path=file_path,
                            file_size=0,
                            create_time=int(time.time()),
                            last_access=int(time.time()),
                            expire_time=expire_time,
                            replica_locations=[b"" for _ in range(3)],
                            status=1,
                            schema_version=1,
                            ext_flags=0,
                            ext_data=b"",
                            ext_data_len=0,
                        )
                        self._v1_meta.put_metadata(meta)
                        flushed_any = True
                        # 移除已刷项
                        bucket.pop(lid, None)
                    except Exception:
                        self._logger.exception("layered flush failed: req=%s lid=%s", plan.req_id, lid)
                if not bucket:
                    self._pending_store.pop(key, None)
                return flushed_any
            else:
                # 聚合写：需要所有层
                num_layers = self._count_layers(fc)
                if len(bucket) < max(1, num_layers):
                    return False
                # 按层序堆叠
                ks, vs = [], []
                for lid in range(num_layers):
                    item = bucket.get(lid)
                    if item is None:
                        return False
                    ks.append(item['k'].unsqueeze(0))
                    vs.append(item['v'].unsqueeze(0))
                k_all = torch.cat(ks, dim=0)
                v_all = torch.cat(vs, dim=0)
                payload_meta = {
                    'schema_version': 1,
                    'tokens_hash': self._tensor_hash(tokens),
                    'num_layers': int(k_all.shape[0]) if k_all.dim() >= 2 else 0,
                    'kv_dtype': str(k_all.dtype),
                    'kv_tail_shape': list(k_all.shape[2:]) if k_all.dim() >= 3 else [],
                    'slots_len': int(slot.numel()),
                    'token_offset': int(start),
                    'block_size': int(end - start),
                }
                data = self._v1_storage.pack_full_payload(k_all, v_all, tokens, torch.ones_like(tokens, dtype=torch.bool), slot, payload_meta)
                file_path = self._make_key(tokens, sess, 0)
                ok = self._v1_storage.upload(file_path, data)
                if not ok:
                    return False
                expire_time = int(getattr(getattr(self.vllm_config, 'kv_transfer_config', SimpleNamespace()), 'kv_expire_time', 86400) or 86400)
                meta = KVMetadata(
                    session_id=sess[:16].ljust(16, b"\x00"),
                    layer_id=0,
                    token_idx=f"{start}-{end}",
                    file_path=file_path,
                    file_size=len(data),
                    create_time=int(time.time()),
                    last_access=int(time.time()),
                    expire_time=expire_time,
                    replica_locations=[b"" for _ in range(3)],
                    status=1,
                    schema_version=1,
                    ext_flags=0,
                    ext_data=b"",
                    ext_data_len=0,
                )
                self._v1_meta.put_metadata(meta)
                # 清理 pending
                self._pending_store.pop(key, None)
                return True
        except Exception:
            self._logger.exception("_flush_pending_span failed")
            return False
