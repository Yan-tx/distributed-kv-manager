from __future__ import annotations

from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage
from distributed_kv_manager.storage.factory import StorageFactory
from distributed_kv_manager.storage.v1.layout import (
    pack_layer_kv_v1,
    unpack_layer_kv_v1,
)


class V1Storage:
    """Thin wrapper on top of an AbstractStorage implementation.

    workflow: upload/download and pack/unpack helper utilities.
    """

    def __init__(self, backend: AbstractStorage) -> None:
        self._backend = backend

    # Basic IO
    def upload(self, file_path: str, data: bytes) -> bool:
        return self._backend.upload(file_path, data)

    def download(self, file_path: str) -> Optional[bytes]:
        return self._backend.download(file_path)

    def exists(self, file_path: str) -> bool:
        return self._backend.exists(file_path)

    # ----- V1 single-layer helpers -----
    def pack_layer_payload(
        self,
        kv_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        slot_mapping: torch.Tensor,
        payload_meta: Optional[dict] = None,
    ) -> bytes:
        """打包单层 KV（形状与 engine 内部保持一致）。"""
        return pack_layer_kv_v1(
            kv_cache=kv_cache,
            input_tokens=input_tokens,
            slot_mapping=slot_mapping,
            payload_meta=payload_meta,
        )

    def unpack_layer_payload(self, data: bytes) -> Tuple[Optional[torch.Tensor], dict]:
        """从单层 payload 中恢复 kv_cache 及元信息。"""
        return unpack_layer_kv_v1(data)

    # ----- Legacy full-payload helpers (保留兼容性，当前 v1 引擎不直接使用) -----
    def pack_full_payload(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        payload_meta: Optional[dict] = None,
    ) -> bytes:
        return self._backend.pack_full_payload(
            k_cache, v_cache, input_tokens, roi, slot_mapping, payload_meta
        )

    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._backend.unpack_kv_data(data)

    def extract_payload_info(self, data: bytes) -> dict:
        return self._backend.extract_payload_info(data)


def create_v1_storage(config) -> V1Storage:
    """Create a V1Storage by delegating to the project's StorageFactory.

    The `config` is expected to have `kv_transfer_config` fields compatible
    with the existing StorageFactory.
    """
    backend = StorageFactory.create_storage(config)
    return V1Storage(backend)
