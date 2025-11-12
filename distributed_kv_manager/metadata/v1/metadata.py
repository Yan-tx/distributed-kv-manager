from __future__ import annotations

from typing import List, Optional

from distributed_kv_manager.metadata.etcd import KVMetadata, KVMetadataManager
from distributed_kv_manager.metadata.metadata_cache import MetadataCache


class V1MetadataClient:
    """A small facade over KVMetadataManager + MetadataCache.

    Provides a simple API for the connector: get/put metadata and
    scan entries filtered by session/layer. Handles async flush via
    the underlying MetadataCache.
    """

    def __init__(self, manager: KVMetadataManager) -> None:
        self._manager = manager
        self._cache = MetadataCache(meta_manager=self._manager)

    def get_metadata(self, key: str, layer_id: Optional[int] = None, session_id: Optional[bytes] = None) -> Optional[KVMetadata]:
        return self._cache.get_metadata(key=key, layer_id=layer_id, session_id=session_id)

    def put_metadata(self, meta: KVMetadata) -> None:
        self._cache.put_metadata(meta)

    def update_access_time(self, key: str) -> None:
        self._cache.update_access_time(key)

    def scan_by_session_layer(self, session_id: Optional[bytes], layer_id: Optional[int]) -> List[KVMetadata]:
        """Return committed metadata entries for a session/layer pair.

        The returned list may be empty if the manager backend does not
        provide listing support.
        """
        out: List[KVMetadata] = []
        try:
            full_keys = self._manager.scan_all_metadata_keys()
        except Exception:
            full_keys = []
        prefix = getattr(self._manager, "prefix", "/kvmeta")
        sid = session_id
        lid = layer_id
        for fk in full_keys:
            try:
                rel = fk
                if isinstance(rel, str) and rel.startswith(prefix + "/"):
                    rel = rel[len(prefix) + 1 :]
                m = self._manager.get_metadata_by_full_key(fk)
                if m is None or getattr(m, "status", None) != 1:
                    continue
                if sid is not None and getattr(m, "session_id", None) != sid:
                    continue
                if lid is not None and getattr(m, "layer_id", None) != lid:
                    continue
                out.append(m)
            except Exception:
                continue
        return out

    def stop(self) -> None:
        try:
            self._cache.stop()
        except Exception:
            pass

