from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


class V1KVEngineImpl(KVConnectorBase_V1):
    """Implement v1 connector接口，但逻辑驻留在 engine/v1 层。

    - Worker 侧：接收 forward_context/kv_caches，调用底层引擎存取。
    - Scheduler 侧：最小实现，当前不做异步传输计划。
    - 存取格式与 kv_engine 一致（文件头、ETCD 元数据、slot_mapping、payload_meta）。
    """

    def __init__(self, vllm_config: Any, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        self._logger = logging.getLogger(self.__class__.__name__)
        # 初始化底层存储/元数据引擎（复用既有实现）
        self._engine = init_engine(vllm_config)
        # 运行时状态
        self._kv_caches: Dict[str, torch.Tensor] = {}
        self._last_forward_context: Any = None
        # 每个请求已保存到的 token 边界
        self._req_last_stored: Dict[str, int] = {}

    # ---------------- Worker-side ----------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._kv_caches = kv_caches or {}

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        # 对首步（num_computed_tokens==0）的请求尝试完整检索
        try:
            reqs = getattr(forward_context, "requests", {}) or {}
            for req_id, req in list(reqs.items()):
                try:
                    num_comp = int(getattr(req, "num_computed_tokens", 0) or 0)
                except Exception:
                    num_comp = 0
                if num_comp == 0:
                    self._retrieve_full(forward_context, req)
        except Exception:
            self._logger.exception("start_load_kv unexpected failure")

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
            reqs = getattr(fc, "requests", {}) or {}
            for req_id, req in list(reqs.items()):
                total = int(getattr(req, "num_computed_tokens", 0) or 0)
                prev = int(self._req_last_stored.get(req_id, 0))
                if total > prev:
                    self._store_slice(fc, req, prev, total)
                    self._req_last_stored[req_id] = total
        except Exception:
            self._logger.exception("wait_for_save unexpected failure")

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # 同步实现且不占用异步队列：不要回传任何已完成ID
        # 否则调度器会尝试对未登记的请求做二次释放，触发 KeyError。
        return set(), set()

    # ---------------- Scheduler-side ----------------
    def get_num_new_matched_tokens(self, request: "Any", num_computed_tokens: int) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(self, request: "Any", blocks: "Any", num_external_tokens: int):
        # 当前无需状态
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        try:
            self._last_forward_context = getattr(scheduler_output, "forward_context", None)
        except Exception:
            self._last_forward_context = None
        return KVConnectorMetadata()

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
        retrieve_kv(fc.model, mi, kvs, rs)
        self._logger.info("[retrieve] req=%s tokens=%d", getattr(req, "request_id", "?"), mi.input_tokens.shape[0])

    def _store_slice(self, fc: Any, req: Any, start: int, end: int) -> None:
        if end <= start:
            return
        mi = self._build_model_input(fc, req, (start, end))
        if mi is None:
            return
        ss = should_store(mi)
        kvs = self._collect_kv(fc, (start, end))
        store_kv(self.vllm_config.model_config, self.vllm_config.parallel_config, None, fc.model, mi, kvs, ss, None)
        self._logger.info("[store] req=%s span=[%d,%d) len=%d", getattr(req, "request_id", "?"), start, end, end-start)

    def _build_model_input(self, fc: Any, req: Any, span: Optional[Tuple[int, int]]) -> Optional[SimpleNamespace]:
        inp = getattr(req, "input_ids", None)
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
