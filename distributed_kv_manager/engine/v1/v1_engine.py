from __future__ import annotations

import logging
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
from distributed_kv_manager.engine import (
    init_engine,
    destroy_engine,
    should_store,
    store_kv,
    should_retrieve,
    retrieve_kv,
)
from distributed_kv_manager.engine.base import RetrieveStatus

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


class _ReqSaveSpec:
    def __init__(self, skip_leading_tokens: int, can_save: bool):
        self.skip_leading_tokens = skip_leading_tokens
        self.can_save = can_save


class _ReqMeta:
    def __init__(self, req_id: str, token_ids: List[int], slot_mapping: torch.Tensor,
                 load_spec: Optional[_ReqLoadSpec], save_spec: Optional[_ReqSaveSpec],
                 is_last_prefill: bool):
        self.req_id = req_id
        self.token_ids = token_ids
        self.slot_mapping = slot_mapping
        self.load_spec = load_spec
        self.save_spec = save_spec
        self.is_last_prefill = is_last_prefill


class _V1Metadata(KVConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: List[_ReqMeta] = []
        # lookup requests to unpin (兼容未来扩展)
        self.lookup_requests_in_step: List[str] = []
    def add_request(self, m: _ReqMeta):
        self.requests.append(m)


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
        # 初始化底层存储/元数据引擎（复用既有实现）
        self._engine = init_engine(vllm_config)
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
        # 保存当前步骤构造的元数据供 worker 使用
        self._current_metadata: Optional[_V1Metadata] = None
        # chunk 大小（按 LMCache 语义）用于决定保存时机；默认 256，可由 kv_transfer_config.chunk_size 覆盖
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
            st_dir = getattr(self._engine, "storage_dir", None)
        except Exception:
            st_dir = None
        try:
            eps = getattr(getattr(self._engine, "_meta", None), "endpoints", None)
        except Exception:
            eps = None
        self._logger.info("[v1_engine] initialized storage_dir=%s etcd_endpoints=%s", st_dir, eps)

    # ---------------- Worker-side ----------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches or {}
        self._logger.info("[v1_engine] registered kv layers=%d", len(self._kv_caches))

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        # 持有 forward_context 供后续 wait_for_save 使用
        self._last_forward_context = forward_context
        # 使用构建好的 metadata 中的 load_spec 执行检索
        meta = self._current_metadata
        if meta is None:
            print("[v1_engine/PRINT] start_load_kv: no metadata present")
            return
        self._logger.info("[v1_engine] start_load_kv: meta.requests=%d", len(meta.requests))
        print(f"[v1_engine/PRINT] start_load_kv using metadata; reqs={len(meta.requests)}")
        for rm in meta.requests:
            ls = rm.load_spec
            if ls is None or not ls.can_load:
                continue
            # 当前实现仅支持全量命中时的加载（kv_engine 只对完整 token 序列命中）
            self._logger.info("[v1_engine] load req=%s cached_tokens=%d vllm_cached=%d", rm.req_id, ls.cached_tokens, ls.vllm_cached_tokens)
            print(f"[v1_engine/PRINT] load attempt req={rm.req_id} cached={ls.cached_tokens} vllm_cached={ls.vllm_cached_tokens}")
            # 如果 vLLM 已有部分（num_computed_tokens>0）则不再检索
            if ls.vllm_cached_tokens == 0 and ls.cached_tokens > 0:
                # 构造一个伪 request 对象：从 unfinished map 获取原始
                req = self._unfinished_requests.get(rm.req_id)
                if req is not None:
                    self._retrieve_full(forward_context, req)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs) -> None:
        # 简化：统一在 wait_for_save 聚合存储
        return

    def wait_for_save(self) -> None:
        try:
            fc = self._last_forward_context
            if fc is None:
                print("[v1_engine/PRINT] wait_for_save: no forward_context captured")
                return
            meta = self._current_metadata
            if meta is None:
                print("[v1_engine/PRINT] wait_for_save: no metadata")
                return
            print(f"[v1_engine/PRINT] wait_for_save called; meta_reqs={len(meta.requests)}")
            for rm in meta.requests:
                req = self._unfinished_requests.get(rm.req_id)
                if req is None:
                    continue
                total_tokens = int(getattr(req, 'num_computed_tokens', 0) or 0)
                if self._skip_last_n > 0 and rm.is_last_prefill:
                    total_tokens = max(0, total_tokens - self._skip_last_n)
                prev = int(self._req_last_stored.get(rm.req_id, 0))
                # 保存策略：达到 chunk 边界 或 last prefill
                boundary = (prev // self._chunk_size + 1) * self._chunk_size
                should_flush = rm.is_last_prefill or total_tokens >= boundary
                if should_flush and total_tokens > prev:
                    self._logger.info("[v1_engine] store req=%s span=[%d,%d) last_prefill=%s", rm.req_id, prev, total_tokens, rm.is_last_prefill)
                    print(f"[v1_engine/PRINT] store flush req={rm.req_id} prev={prev} total={total_tokens} last={rm.is_last_prefill}")
                    self._store_slice(fc, req, prev, total_tokens)
                    self._req_last_stored[rm.req_id] = total_tokens
            # 可选：同步等待写入完成，方便立即看到落盘
            try:
                vcfg = getattr(self.vllm_config, "kv_transfer_config", None)
                force_sync = bool(getattr(vcfg, "force_sync_store", False))
            except Exception:
                force_sync = False
            if force_sync:
                futures = list(getattr(self._engine, "_futures", []) or [])
                self._logger.info("[v1_engine] force_sync_store waiting %d futures", len(futures))
                print(f"[v1_engine/PRINT] force_sync_store waiting futures={len(futures)}")
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
        """调度侧：返回需要额外分配的外部缓存命中 token 数（语义对齐 LMCache）。

        当前后端只能在完整 prompt 命中时返回其长度（需回退最后1 token 重新计算）。
        """
        # 构造 model_input 以调用 should_retrieve
        mi = self._build_model_input(SimpleNamespace(model=request.model if hasattr(request, 'model') else None), request, None)
        if mi is None:
            return 0, False
        rs = should_retrieve(mi)
        prompt_len = int(getattr(request, 'num_tokens', len(getattr(request, 'prompt_token_ids', []))) or 0)
        if isinstance(rs, RetrieveStatus) and rs == RetrieveStatus.HIT:
            cached = prompt_len
            need_allocate = max(0, cached - num_computed_tokens - 1)  # 重新计算最后一个 token
            self._load_specs[getattr(request, 'request_id', getattr(request, 'req_id', ''))] = _ReqLoadSpec(num_computed_tokens, cached, need_allocate > 0)
            self._logger.info("[v1_engine] match req=%s cached=%d computed=%d need=%d", getattr(request, 'request_id', '?'), cached, num_computed_tokens, need_allocate)
            return need_allocate, True
        else:
            self._load_specs[getattr(request, 'request_id', getattr(request, 'req_id', ''))] = _ReqLoadSpec(num_computed_tokens, 0, False)
            return 0, False

    def update_state_after_alloc(self, request: "Any", blocks: "Any", num_external_tokens: int):
        # 记录 unfinished 请求供后续 build_connector_meta/start_load_kv 使用
        req_id = getattr(request, 'request_id', getattr(request, 'req_id', ''))
        self._unfinished_requests[req_id] = request
        # 维护 token id 序列（prompt token + 已调度新 token）简化：直接使用 prompt_token_ids
        toks = list(getattr(request, 'prompt_token_ids', []))
        self._request_tokens[req_id] = toks
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
            num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
            is_last_prefill = (num_comp + num_sched) >= len(toks)
            # 保存策略：初始 skip_leading 为已持久化 token 数（上次为 0）
            save_spec = _ReqSaveSpec(skip_leading_tokens=self._req_last_stored.get(req_id, 0), can_save=True)
            slot_mapping = torch.arange(len(toks), dtype=torch.long)
            meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
            # 保存 tracker
            self._request_tokens[req_id] = toks
            self._unfinished_requests[req_id] = req

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
                num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                is_last_prefill = (num_comp + num_sched) >= len(toks)
                save_spec = _ReqSaveSpec(skip_leading_tokens=self._req_last_stored.get(req_id, 0), can_save=True)
                slot_mapping = torch.arange(len(toks), dtype=torch.long)
                meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
        else:
            try:
                for i, req_id in enumerate(getattr(cached_reqs, 'req_ids', []) or []):
                    req = self._unfinished_requests.get(req_id)
                    toks = self._request_tokens.get(req_id, list(getattr(req, 'prompt_token_ids', [])))
                    num_comp = int(getattr(req, 'num_computed_tokens', 0) or 0)
                    load_spec = None
                    num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                    is_last_prefill = (num_comp + num_sched) >= len(toks)
                    save_spec = _ReqSaveSpec(skip_leading_tokens=self._req_last_stored.get(req_id, 0), can_save=True)
                    slot_mapping = torch.arange(len(toks), dtype=torch.long)
                    meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
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
    def _retrieve_full(self, fc: Any, req: Any) -> None:
        mi = self._build_model_input(fc, req, None)
        if mi is None:
            return
        rs = should_retrieve(mi)
        kvs = self._collect_kv(fc, None)
        if not kvs:
            self._logger.warning("[v1_engine] retrieve: no kv_caches available (registered=%d fc_has=%s)",
                                 len(self._kv_caches), hasattr(fc, "kv_caches"))
            print(f"[v1_engine/PRINT] retrieve_full: NO kv_caches registered={len(self._kv_caches)} fc_has_kv={hasattr(fc,'kv_caches')}")
        retrieve_kv(fc.model, mi, kvs, rs)
        self._logger.info("[retrieve] req=%s tokens=%d", getattr(req, "request_id", "?"), mi.input_tokens.shape[0])
        print(f"[v1_engine/PRINT] retrieve_full done req={getattr(req,'request_id','?')} tokens={mi.input_tokens.shape[0]} kv_layers={len(kvs)}")

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
            print(f"[v1_engine/PRINT] store_slice: NO kv_caches req={getattr(req,'request_id','?')} span=[{start},{end})")
        store_kv(self.vllm_config.model_config, self.vllm_config.parallel_config, None, fc.model, mi, kvs, ss, None)
        self._logger.info("[store] req=%s span=[%d,%d) len=%d", getattr(req, "request_id", "?"), start, end, end-start)
        print(f"[v1_engine/PRINT] store_slice done req={getattr(req,'request_id','?')} span=[{start},{end}) len={end-start} kv_layers={len(kvs)}")

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
        am.slot_mapping = torch.arange(tokens.shape[0], device=dev)
        mi.attn_metadata = am
        try:
            sid_src = getattr(getattr(self.vllm_config, "kv_transfer_config", None), "engine_id", None)
            sid = (str(sid_src) if sid_src else str(getattr(req, "request_id", "v1_session"))).encode("utf-8")
        except Exception:
            sid = b"v1_session"
        mi.session_id = sid
        mi.layer_id = 0
        if span is not None:
            mi.payload_meta = {"token_offset": int(span[0]), "block_size": int(tokens.shape[0])}
        return mi

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
        try:
            return next(fc.model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        destroy_engine()
    finally:
        _CORE_SINGLETON = None


def v1_should_store(model_input: Any) -> Any:
    # 直接沿用底层策略
    return should_store(model_input)


def v1_store_kv(model_config: Any, parallel_config: Any, sampler: Any, model_executable: Any,
                model_input: Any, kv_caches: List[torch.Tensor], store_status: Any,
                hidden_states: Optional[torch.Tensor]) -> None:
    return store_kv(model_config, parallel_config, sampler, model_executable, model_input, kv_caches, store_status, hidden_states)


def v1_should_retrieve(model_input: Any) -> Any:
    return should_retrieve(model_input)


def v1_retrieve_kv(model_executable: Any, model_input: Any,
                   kv_caches: List[torch.Tensor], retrieve_status: Any) -> Any:
    return retrieve_kv(model_executable, model_input, kv_caches, retrieve_status)
