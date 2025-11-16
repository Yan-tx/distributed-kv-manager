from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import safetensors
import torch

from .base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

try:
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore
    logger = _vllm_init_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


# ---------------- Data structures  ----------------


@dataclass
class _ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "_ReqMeta":
        valid_num_tokens = _align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = int(block_ids_tensor.shape[0])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return _ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class _V1Metadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            _ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


# ---------------- Engine (connector) implementation ----------------


class V1KVEngineImpl(KVConnectorBase_V1):
    """V1 引擎

    - 调度：判断 External Cache 命中、构建 load/store 元数据计划。
    - Worker：按 slot_mapping 将 safetensors 中的 KV 注入/提取。
    - 存储：使用本地共享目录（忽略本项目的 ETCD/Storage）。
    """

    def __init__(self, vllm_config: Any, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        try:
            bs = getattr(getattr(vllm_config, 'cache_config', None), 'block_size', 16)
        except Exception:
            bs = 16
        self._block_size = int(bs)
        self._requests_need_load: dict[str, Any] = {}
        self._storage_path = self._resolve_dkv_path(vllm_config) or "/tmp/kvcache/v1"
        logger.info("[v1_engine] dkv_storage_path=%s", self._storage_path)

    # ---------------- Worker-side ----------------
    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        metadata: _V1Metadata = self._get_connector_metadata()  # type: ignore
        if metadata is None or not isinstance(metadata, _V1Metadata):
            logger.warning("[v1_engine] start_load_kv: connector metadata is None")
            return
        attn_metadata = getattr(forward_context, 'attn_metadata', None)
        if attn_metadata is None:
            logger.warning("[v1_engine] start_load_kv: attn_metadata is None")
            return
        layers = getattr(forward_context, 'no_compile_layers', {})
        for req in metadata.requests:
            if req.is_store:
                continue
            logger.info("[v1_engine] inject %d tokens into paged KV", int(req.slot_mapping.numel()))
            for layer_name in layers:
                layer = layers[layer_name]
                kv_cache_attr = getattr(layer, 'kv_cache', None)
                if kv_cache_attr is None:
                    continue
                kv_cache_layer = kv_cache_attr[getattr(forward_context, 'virtual_engine', 0)]
                filename = self._generate_filename_debug(layer_name, req.token_ids, req.mm_hashes)
                try:
                    kv_cache = safetensors.torch.load_file(filename)["kv_cache"].cuda()
                except Exception:
                    continue
                _inject_kv_into_layer(kv_cache_layer, kv_cache, req.slot_mapping, attn_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        metadata: _V1Metadata = self._get_connector_metadata()  # type: ignore
        if metadata is None or not isinstance(metadata, _V1Metadata):
            return
        for req in metadata.requests:
            if not req.is_store:
                continue
            filename = self._generate_filename_debug(layer_name, req.token_ids, req.mm_hashes)
            kv_cache = _extract_kv_from_layer(kv_layer, req.slot_mapping, attn_metadata)
            tensors = {"kv_cache": kv_cache.detach().cpu()}
            try:
                safetensors.torch.save_file(tensors, filename)
            except Exception:
                pass

    def wait_for_save(self):
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    # ---------------- Scheduler-side ----------------
    def get_num_new_matched_tokens(self, request: "Any", num_computed_tokens: int) -> tuple[int, bool]:
        # 判断共享目录下是否存在命中目录
        try:
            token_ids = list(getattr(request, 'prompt_token_ids', []) or [])
            mm_hashes = [getattr(f, 'identifier', '') for f in (getattr(request, 'mm_features', []) or [])]
            num_tokens_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
            folder = self._generate_foldername_debug(torch.tensor(token_ids)[:num_tokens_to_check], mm_hashes, create_folder=False)
            if not os.path.exists(folder):
                return 0, False
            logger.info("External Cache Hit!")
            return max(0, num_tokens_to_check - int(num_computed_tokens)), False
        except Exception:
            return 0, False

    def update_state_after_alloc(self, request: "Any", blocks: "Any", num_external_tokens: int):
        if int(num_external_tokens or 0) > 0:
            self._requests_need_load[getattr(request, 'request_id', '')] = request
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        meta = _V1Metadata()
        # 新调度请求
        for new_req in getattr(scheduler_output, 'scheduled_new_reqs', []) or []:
            token_ids = list(getattr(new_req, 'prompt_token_ids', []) or [])
            mm_hashes = [getattr(f, 'identifier', '') for f in (getattr(new_req, 'mm_features', []) or [])]
            if getattr(new_req, 'req_id', None) in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=(getattr(new_req, 'block_ids', [[0]]) or [[0]])[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
            else:
                num_tokens_to_check = _align_to_block_size(len(token_ids) - 1, self._block_size)
                folder = self._generate_foldername_debug(torch.tensor(token_ids)[:num_tokens_to_check], mm_hashes, create_folder=False)
                if not os.path.exists(folder):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=(getattr(new_req, 'block_ids', [[0]]) or [[0]])[0],
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )

        # 被恢复的 cached 请求（可选）
        cached_reqs = getattr(scheduler_output, 'scheduled_cached_reqs', None)
        try:
            req_ids = list(getattr(cached_reqs, 'req_ids', []) or [])
            resumed = list(getattr(cached_reqs, 'resumed_from_preemption', []) or [])
            new_block_ids = list(getattr(cached_reqs, 'new_block_ids', []) or [])
            num_computed = list(getattr(cached_reqs, 'num_computed_tokens', []) or [])
        except Exception:
            req_ids, resumed, new_block_ids, num_computed = [], [], [], []
        for i, req_id in enumerate(req_ids):
            if not (resumed[i] if i < len(resumed) else False):
                break
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                total_tokens = int((num_computed[i] if i < len(num_computed) else 0) + getattr(scheduler_output, 'num_scheduled_tokens', {}).get(req_id, 0))
                token_ids = list(getattr(request, 'all_token_ids', [])[:total_tokens])
                nbids = new_block_ids[i] if i < len(new_block_ids) else [[0]]
                block_ids = nbids[0] if nbids else [0]
                mm_hashes = [getattr(f, 'identifier', '') for f in (getattr(request, 'mm_features', []) or [])]
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
        self._requests_need_load.clear()
        return meta

    # ---------------- helpers ----------------
    def _resolve_dkv_path(self, vllm_config: Any) -> Optional[str]:
        try:
            kvt = getattr(vllm_config, 'kv_transfer_config', None)
            if kvt is None:
                return None
            path = getattr(kvt, 'dkv_storage_path', None)
            if path:
                return str(path)
            extra = getattr(kvt, 'extra_config', None)
            if isinstance(extra, dict):
                p = extra.get('dkv_storage_path')
                if p:
                    return str(p)
        except Exception:
            pass
        return None

    def _generate_foldername_debug(self, token_ids: torch.Tensor, mm_hashes: list[str], create_folder: bool = False) -> str:
        try:
            token_bytes = token_ids.cpu().numpy().tobytes()
        except Exception:
            token_bytes = bytes()
        if mm_hashes:
            try:
                token_bytes += "-".join(mm_hashes).encode('utf-8')
            except Exception:
                pass
        try:
            import hashlib as _hash
            input_ids_hash = _hash.md5(token_bytes, usedforsecurity=False).hexdigest()
        except Exception:
            input_ids_hash = ""
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            try:
                os.makedirs(foldername, exist_ok=True)
            except Exception:
                pass
        return foldername

    def _generate_filename_debug(self, layer_name: str, token_ids: torch.Tensor, mm_hashes: list[str]) -> str:
        foldername = self._generate_foldername_debug(token_ids, mm_hashes, create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")


# ---------------- functional helpers  ----------------


def _inject_kv_into_layer(dst_kv_cache_layer: torch.Tensor, src_kv_cache: torch.Tensor, slot_mapping: torch.Tensor, attn_metadata: Any) -> None:
    shape = dst_kv_cache_layer.shape
    try:
        from vllm.v1.attention.backends.mla.common import MLACommonMetadata as _MLA
    except Exception:  # pragma: no cover
        _MLA = object  # type: ignore
    if isinstance(attn_metadata, _MLA):
        num_pages, page_size = int(shape[0]), int(shape[1])
        view = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
        view[slot_mapping, ...] = src_kv_cache
        view.reshape(shape)
    else:
        num_pages, page_size = int(shape[1]), int(shape[2])
        view = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        view[:, slot_mapping, ...] = src_kv_cache
        view.reshape(shape)


def _extract_kv_from_layer(layer: torch.Tensor, slot_mapping: torch.Tensor, attn_metadata: Any) -> torch.Tensor:
    try:
        from vllm.v1.attention.backends.mla.common import MLACommonMetadata as _MLA
    except Exception:  # pragma: no cover
        _MLA = object  # type: ignore
    if isinstance(attn_metadata, _MLA):
        num_pages, page_size = layer.shape[0], layer.shape[1]
        return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
    num_pages, page_size = layer.shape[1], layer.shape[2]
    return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens - 1) // block_size * block_size


# ---------------- Module-level helpers (singleton) ----------------

_CORE_SINGLETON: Optional[V1KVEngineImpl] = None


def init_v1_engine(vllm_config: Any, role: Any = None) -> V1KVEngineImpl:
    global _CORE_SINGLETON
    if _CORE_SINGLETON is None:
        real_role = role if role is not None else KVConnectorRole.WORKER
        _CORE_SINGLETON = V1KVEngineImpl(vllm_config, real_role)
    return _CORE_SINGLETON


def destroy_v1_engine():
    global _CORE_SINGLETON
    _CORE_SINGLETON = None


def v1_should_store(model_input: Any) -> Any:
    return None


def v1_store_kv(model_config: Any, parallel_config: Any, sampler: Any, model_executable: Any,
                model_input: Any, kv_caches: list[torch.Tensor], store_status: Any,
                hidden_states: Optional[torch.Tensor]) -> None:
    return None


def v1_should_retrieve(model_input: Any) -> Any:
    return None


def v1_retrieve_kv(model_executable: Any, model_input: Any,
                   kv_caches: list[torch.Tensor], retrieve_status: Any) -> Any:
    return None

