from __future__ import annotations

from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage
from distributed_kv_manager.storage.factory import StorageFactory


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

    # Payload helpers (full payload preferred)
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

