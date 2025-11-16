from __future__ import annotations

from typing import List, Optional

from distributed_kv_manager.metadata.etcd import KVMetadata, KVMetadataManager
from distributed_kv_manager.metadata.metadata_cache import MetadataCache


class V1MetadataClient:
    """Facade over KVMetadataManager + MetadataCache for v1 external KV.

    v1 只需要「某个 hash 目录是否存在有效 KV」这一粒度，因此这里约定：
    - file_path/key: 使用完整的 hash 目录绝对路径（例如 /kvcache_v1/index/<hash>）
    - layer_id: v1 聚合 entry 使用 -1 作为占位
    - session_id: 暂时使用常量占位，后续可与真实会话绑定
    """

    def __init__(self, manager: KVMetadataManager, default_expire: int = 0) -> None:
        self._manager = manager
        self._cache = MetadataCache(meta_manager=self._manager)
        self._default_expire = int(default_expire or 0)

    def get_metadata(
        self,
        key: str,
        layer_id: Optional[int] = None,
        session_id: Optional[bytes] = None,
    ) -> Optional[KVMetadata]:
        return self._cache.get_metadata(key=key, layer_id=layer_id, session_id=session_id)

    def put_metadata(self, meta: KVMetadata) -> None:
        self._cache.put_metadata(meta)

    def update_access_time(self, key: str) -> None:
        self._cache.update_access_time(key)

    # ---- v1 专用 helper：按 hash 目录维度判断 / 标记存储 ----

    def hash_exists(self, folder_abs: str) -> bool:
        """判断某个 hash 目录是否存在有效 KV（未过期且 status==1）。"""
        meta = self.get_metadata(key=folder_abs)
        if meta is None or getattr(meta, "status", 0) != 1:
            return False
        if meta.is_expired():
            return False
        # 触发一次 access time 更新（异步写入）
        self.update_access_time(folder_abs)
        return True

    def mark_hash_stored(
        self,
        folder_abs: str,
        num_tokens: int,
        file_size: int,
    ) -> None:
        """为某个 hash 目录写入一条聚合元数据记录。"""
        import time as _time

        now = int(_time.time())
        expire = int(self._default_expire or 0)
        session_id = b"v1_external_kv__"  # 16B 占位
        layer_id = -1
        token_idx = str(int(num_tokens))
        meta = KVMetadata(
            session_id=session_id,
            layer_id=layer_id,
            token_idx=token_idx,
            file_path=folder_abs,
            file_size=int(file_size),
            create_time=now,
            last_access=now,
            expire_time=expire,
            replica_locations=[b"", b"", b""],
            status=1,
            schema_version=1,
            ext_flags=0,
            ext_data=b"",
            ext_data_len=0,
        )
        self.put_metadata(meta)

    # ---- 通用扫描接口（目前 v1 未使用，可保留） ----

    def scan_by_session_layer(
        self, session_id: Optional[bytes], layer_id: Optional[int]
    ) -> List[KVMetadata]:
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
