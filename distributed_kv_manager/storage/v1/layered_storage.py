from __future__ import annotations

from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage


class LayeredV1Storage:
    """Per-layer storage facade for v1 connector.

    Stores each layer as an independent payload file. The filename encodes
    session, layer, and a hash of the token slice to avoid collisions.

    Payload format reuses AbstractStorage.pack_full_payload with num_layers=1.
    """

    def __init__(self, backend: AbstractStorage) -> None:
        self._backend = backend

    @staticmethod
    def _hash_tokens(tokens: torch.Tensor) -> str:
        import hashlib
        if tokens.numel() == 0:
            return "empty"
        return hashlib.blake2b(tokens.cpu().numpy().tobytes(), digest_size=24).hexdigest()

    def _file_name(self, session_id: bytes, layer_id: int, tokens: torch.Tensor, start: int, end: int) -> str:
        sess_str = session_id.decode('utf-8', errors='ignore') if isinstance(session_id, (bytes, bytearray)) else str(session_id)
        h = self._hash_tokens(tokens)
        # keep short to fit metadata file_path limit (128 bytes)
        return f"kv_{sess_str}_L{int(layer_id)}_{start}-{end}_{h}.pt"

    def upload_layer_slice(
        self,
        session_id: bytes,
        layer_id: int,
        tokens: torch.Tensor,
        k_slice: torch.Tensor,
        v_slice: torch.Tensor,
        slot: torch.Tensor,
        start: int,
        end: int,
        payload_meta_extra: Optional[dict] = None,
    ) -> Tuple[str, int]:
        # normalize to [1, seq, ...]
        if k_slice.dim() >= 1 and k_slice.shape[0] != 1:
            k_payload = k_slice.unsqueeze(0)
            v_payload = v_slice.unsqueeze(0)
        else:
            k_payload = k_slice
            v_payload = v_slice

        payload_meta = {
            'schema_version': 1,
            'tokens_hash': self._hash_tokens(tokens),
            'num_layers': 1,
            'kv_dtype': str(k_slice.dtype),
            'kv_tail_shape': list(k_slice.shape[1:]) if k_slice.dim() >= 2 else [],
            'slots_len': int(slot.numel()),
            'token_offset': int(start),
            'block_size': int(end - start),
            'layer_id': int(layer_id),
        }
        if isinstance(payload_meta_extra, dict):
            for k, v in payload_meta_extra.items():
                if k not in payload_meta:
                    payload_meta[k] = v

        roi = torch.ones_like(tokens, dtype=torch.bool)
        data = self._backend.pack_full_payload(k_payload, v_payload, tokens, roi, slot, payload_meta)
        file_path = self._file_name(session_id, layer_id, tokens, start, end)
        ok = self._backend.upload(file_path, data)
        if not ok:
            raise RuntimeError(f"upload failed: {file_path}")
        return file_path, len(data)

    def download_layer_slice(
        self,
        session_id: bytes,
        layer_id: int,
        tokens: torch.Tensor,
        start: int,
        end: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        file_path = self._file_name(session_id, layer_id, tokens, start, end)
        data = self._backend.download(file_path)
        if data is None:
            return None, None, None
        info = self._backend.extract_payload_info(data)
        k, v = self._backend.unpack_kv_data(data)
        slot = None
        try:
            slot = info.get('slot_mapping') if isinstance(info, dict) else None
        except Exception:
            slot = None
        return k, v, slot

