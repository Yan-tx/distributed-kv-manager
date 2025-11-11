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
        # 若 vllm_config.kv_transfer_config.config_path 存在，则传递以确保测试/自定义配置覆盖默认config.json
        try:
            cfg_path = getattr(getattr(vllm_config, 'kv_transfer_config', None), 'config_path', None)
        except Exception:
            cfg_path = None
        self._engine = init_engine(vllm_config, config_path=cfg_path)
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
            st_dir = getattr(self._engine, "storage_dir", None)
        except Exception:
            st_dir = None
        try:
            eps = getattr(getattr(self._engine, "_meta", None), "endpoints", None)
        except Exception:
            eps = None
        self._logger.info("[v1_engine] initialized storage_dir=%s etcd_endpoints=%s (synchronous load/save)", st_dir, eps)

    # ---------------- Worker-side ----------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches or {}
        self._logger.info("[v1_engine] registered kv layers=%d", len(self._kv_caches))

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        # 持有 forward_context 供后续 wait_for_save 使用
        self._last_forward_context = forward_context
        # 使用构建好的 metadata 中的 load_spec 执行检索
        try:
            meta = self._get_connector_metadata()
        except Exception:
            meta = self._current_metadata
        if meta is None:
            return
        self._logger.debug("[v1_engine] start_load_kv: meta.requests=%d", len(meta.requests))
        for rm in meta.requests:
            ls = rm.load_spec
            if ls is None or not ls.can_load:
                continue
            # 当前实现仅支持全量命中时的加载（kv_engine 只对完整 token 序列命中）
            self._logger.debug("[v1_engine] load req=%s cached_tokens=%d vllm_cached=%d", rm.req_id, ls.cached_tokens, ls.vllm_cached_tokens)
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
                return
            try:
                meta = self._get_connector_metadata()
            except Exception:
                meta = self._current_metadata
            if meta is None:
                return
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
                    self._logger.debug("[v1_engine] store req=%s span=[%d,%d) last_prefill=%s", rm.req_id, prev, total_tokens, rm.is_last_prefill)
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
            self._logger.info("[v1_engine] match req=%s cached=%d computed=%d need=%d (async=False)", getattr(request, 'request_id', '?'), cached, num_computed_tokens, need_allocate)
            # 同步实现：第二返回值（是否异步加载）恒为 False；当 need 为 0 时必须为 False
            return need_allocate, False
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
            num_sched = int(getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
            is_last_prefill = (num_comp + num_sched) >= len(toks)
            # 保存策略：初始 skip_leading 为已持久化 token 数（上次为 0）
            save_spec = _ReqSaveSpec(skip_leading_tokens=self._req_last_stored.get(req_id, 0), can_save=True)
            slot_mapping = self._build_slot_mapping_from_blocks(req_id, len(toks))
            meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
            # 保存 tracker
            self._request_tokens[req_id] = toks
            self._unfinished_requests[req_id] = req
            self._req_slot_mapping[req_id] = slot_mapping

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
                slot_mapping = self._build_slot_mapping_from_blocks(req_id, len(toks))
                meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
                self._req_slot_mapping[req_id] = slot_mapping
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
                    slot_mapping = self._build_slot_mapping_from_blocks(req_id, len(toks))
                    meta.add_request(_ReqMeta(req_id, toks, slot_mapping, load_spec, save_spec, is_last_prefill))
                    self._req_slot_mapping[req_id] = slot_mapping
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
        model_exec = self._get_model_exec(fc)
        if model_exec is None:
            self._logger.warning("[v1_engine] retrieve: forward_context has no model; skip")
            return
        retrieve_kv(model_exec, mi, kvs, rs)
        self._logger.debug("[retrieve] req=%s tokens=%d", getattr(req, "request_id", "?"), mi.input_tokens.shape[0])

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

    def _build_slot_mapping_from_blocks(self, req_id: str, token_len: int) -> torch.Tensor:
        """根据记录的 block_ids 和 vLLM 的 block_size 构造 1D slot_mapping。

        若无可用 block_ids，则回退为 arange。
        """
        try:
            block_ids = self._req_block_ids.get(req_id, [])
            if not block_ids:
                return torch.arange(token_len, dtype=torch.long)
            # 获取 block_size
            bs = None
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
