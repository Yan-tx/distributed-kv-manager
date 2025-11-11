# SPDX-License-Identifier: Apache-2.0
# This module duplicates the v1 connector so it lives inside the distributed_kv_manager package path
# allowing `kv_connector_module_path = "distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1"`

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace
import logging
import torch

try:
    from vllm.config import VllmConfig
    from vllm.logger import init_logger
except Exception:  # pragma: no cover - compatibility shim
    class VllmConfig:  # type: ignore
        def __init__(self) -> None:
            self.model_config = None
            self.parallel_config = None
            self.v1_config = type("X", (), {"gpu_block_size": 16})()
            self.gpu_block_size = 16

    def init_logger(name: str):  # type: ignore
        import logging as _logging
        return _logging.getLogger(name)
try:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
except Exception:  # pragma: no cover - compatibility shim
    # 轻量级兜底 stub，避免不同 vLLM 版本缺失符号导致 ImportError。
    from typing import Any, Dict, List

    class KVConnectorBase_V1:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

    class KVConnectorRole:  # type: ignore
        name = "unknown"

    class KVConnectorMetadata:  # type: ignore
        pass

    class KVCacheConfig:  # type: ignore
        pass

    class Request:  # type: ignore
        def __init__(self) -> None:
            self.request_id = "stub"
            self.num_computed_tokens = 0
            # Some versions use req_id
            self.req_id = self.request_id

    class KVCacheBlocks:  # type: ignore
        def get_block_ids(self):
            return [[0]]

    class _Cached:  # helper for scheduled_cached_reqs
        def __init__(self) -> None:
            self.req_ids: List[str] = []
            self.new_block_ids: List[List[int]] = []

    class SchedulerOutput:  # type: ignore
        def __init__(self) -> None:
            self.scheduled_new_reqs: List[Request] = []
            self.num_scheduled_tokens: Dict[str, int] = {}
            self.scheduled_cached_reqs = _Cached()
            # forward_context 可能存在于部分版本，这里让它自引用以复用字段名
            self.forward_context = self
            # requests map, used by some code paths
            self.requests: Dict[str, Request] = {}

from distributed_kv_manager.engine import (
    init_engine,
    retrieve_kv,
    should_retrieve,
    store_kv,
    should_store,
    destroy_engine,
)

logger = init_logger(__name__)


@dataclass
class DKVTransferItem:
    req_id: str
    start_token: int
    end_token: int


@dataclass
class DKVOffloadingConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    reqs_to_store: dict[str, list[DKVTransferItem]]
    reqs_to_load: dict[str, list[DKVTransferItem]]


class DKVOffloadingConnector(KVConnectorBase_V1):  # type: ignore[misc]
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # vLLM v1 (e.g., 0.9.2) base connector __init__ often takes (config, role) only.
        # Pass only the first two to remain compatible; keep kv_cache_config for forward-compat.
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        self.role = role
        self._engine = init_engine(vllm_config)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("DKVOffloadingConnector initialized (role=%s)", role.name)
        self._requests: dict[str, Request] = {}
        self._request_block_ids: dict[str, list[int]] = {}
        self._planned_end_token: dict[str, int] = {}

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        pass

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        meta = self._ensure_worker_meta()
        if not meta.reqs_to_load:
            return
        for req_id, _items in meta.reqs_to_load.items():
            try:
                req = forward_context.requests.get(req_id)
                if req is None:
                    continue
                self._engine_should_retrieve_and_retrieve_full(forward_context, req)
            except Exception:
                self._logger.exception("load_kv failed for req=%s", req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        meta = self._ensure_worker_meta()
        if not meta.reqs_to_store:
            return
        try:
            fc = self._last_forward_context
        except Exception:
            fc = None
        for req_id, items in meta.reqs_to_store.items():
            try:
                if fc is None:
                    continue
                req = fc.requests.get(req_id)
                if req is None:
                    continue
                for it in items:
                    self._engine_store_slice(fc, req, it.start_token, it.end_token)
            except Exception:
                self._logger.exception("wait_for_save store failed for req=%s", req_id)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        return set(finished_req_ids), set()

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]
        self._request_block_ids[request.request_id] = block_ids

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        reqs_to_store: dict[str, list[DKVTransferItem]] = {}
        reqs_to_load: dict[str, list[DKVTransferItem]] = {}

        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            req = self._requests.get(req_id)
            if req is None:
                continue
            new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if new_tokens <= 0:
                continue
            prev_end = self._planned_end_token.get(req_id, 0)
            end_tok = req.num_computed_tokens + new_tokens
            if end_tok > prev_end:
                reqs_to_store.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, prev_end, end_tok)
                )
                self._planned_end_token[req_id] = end_tok

        cached = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached.req_ids):
            req = self._requests.get(req_id)
            if req is None:
                continue
            new_block_ids_tuple = cached.new_block_ids[idx]
            try:
                gpu_bs = self.vllm_config.v1_config.gpu_block_size
            except Exception:
                gpu_bs = getattr(self.vllm_config, "gpu_block_size", 16)
            if not new_block_ids_tuple:
                continue
            min_blk = min(new_block_ids_tuple)
            max_blk = max(new_block_ids_tuple)
            start_tok = min_blk * gpu_bs
            end_tok = (max_blk + 1) * gpu_bs
            prev_end = self._planned_end_token.get(req_id, 0)
            if end_tok > prev_end:
                reqs_to_store.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, max(prev_end, start_tok), end_tok)
                )
                self._planned_end_token[req_id] = end_tok

        meta = DKVOffloadingConnectorMetadata(
            reqs_to_store=reqs_to_store,
            reqs_to_load=reqs_to_load,
        )
        try:
            self._last_forward_context = scheduler_output.forward_context
        except Exception:
            self._last_forward_context = None
        self._connector_metadata = meta
        return meta

    def update_connector_output(self, connector_output: Any):
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._planned_end_token.pop(req_id, None)
        return False, None

    def take_events(self):
        return []

    def _engine_should_retrieve_and_retrieve_full(self, fc: Any, req: "Request"):
        mi = SimpleNamespace()
        input_ids = getattr(req, "input_ids", None)
        if input_ids is None:
            return
        if not torch.is_tensor(input_ids):
            input_tokens = torch.tensor(input_ids, dtype=torch.long, device=self._infer_device(fc))
        else:
            input_tokens = input_ids
        mi.input_tokens = input_tokens
        sa = SimpleNamespace()
        sa.seq_lens = [int(input_tokens.shape[0])]
        sa.slot_mapping = torch.arange(sa.seq_lens[0], device=input_tokens.device)
        try:
            sid = str(getattr(req, "request_id", "v1_session")).encode("utf-8")
        except Exception:
            sid = b"v1_session"
        mi.session_id = sid
        mi.layer_id = 0
        kv_caches = self._collect_kv_caches(fc)
        rs = should_retrieve(mi)
        retrieve_kv(fc.model, mi, kv_caches, rs)

    def _engine_store_slice(self, fc: Any, req: "Request", start_tok: int, end_tok: int):
        if end_tok <= start_tok:
            return
        input_ids = getattr(req, "input_ids", None)
        if input_ids is None:
            return
        dev = self._infer_device(fc)
        if not torch.is_tensor(input_ids):
            full_tokens = torch.tensor(input_ids, dtype=torch.long, device=dev)
        else:
            full_tokens = input_ids
        curr_tokens = full_tokens[start_tok:end_tok]
        seq_len = int(curr_tokens.shape[0])
        mi = SimpleNamespace()
        mi.input_tokens = curr_tokens
        sa = SimpleNamespace()
        sa.seq_lens = [seq_len]
        sa.slot_mapping = torch.arange(seq_len, device=dev)
        mi.attn_metadata = sa
        try:
            sid = str(getattr(req, "request_id", "v1_session")).encode("utf-8")
        except Exception:
            sid = b"v1_session"
        mi.session_id = sid
        mi.layer_id = 0
        mi.payload_meta = {"token_offset": int(start_tok), "block_size": int(seq_len)}
        kv_caches = self._collect_kv_caches(fc, span=(start_tok, end_tok))
        ss = should_store(mi)
        store_kv(self.vllm_config.model_config, self.vllm_config.parallel_config, None, fc.model, mi, kv_caches, ss, None)

    def _ensure_worker_meta(self) -> DKVOffloadingConnectorMetadata:
        meta = getattr(self, "_connector_metadata", None)
        if isinstance(meta, DKVOffloadingConnectorMetadata):
            return meta
        return DKVOffloadingConnectorMetadata(reqs_to_store={}, reqs_to_load={})

    def _collect_kv_caches(self, fc: Any, span: tuple[int, int] | None = None) -> list[torch.Tensor]:
        try:
            caches_by_layer = fc.kv_caches
            tensors = list(caches_by_layer.values())
            if span is None:
                return tensors
            start, end = span
            out = []
            for t in tensors:
                try:
                    k = t[0]
                    v = t[1]
                    if k.dim() == 4:
                        out.append(torch.stack([k[0:1, start:end].contiguous(), v[0:1, start:end].contiguous()], dim=0))
                    elif k.dim() == 3:
                        out.append(torch.stack([k[start:end].contiguous(), v[start:end].contiguous()], dim=0))
                    else:
                        out.append(t)
                except Exception:
                    out.append(t)
            return out
        except Exception:
            return []

    def _infer_device(self, fc: Any) -> torch.device:
        try:
            device = next(fc.model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def close(self):
        try:
            destroy_engine()
        except Exception:
            pass


class DKVEngineConnectorV1(DKVOffloadingConnector):
    pass
