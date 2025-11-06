import os
import re
from typing import Optional

from .base import AbstractStorage


_HEX_RE = re.compile(r"([0-9a-fA-F]{16,})")


def extract_hash_from_key(key: str) -> Optional[str]:
    """Try to extract a hex hash substring from logical key.

    Heuristic: take the last long hex run (>=16) in the key.
    Returns None if not found.
    """
    m = list(_HEX_RE.finditer(key))
    if not m:
        return None
    return m[-1].group(1).lower()


class HashBucketPathMapper:
    """Map logical keys to a hash-bucketed relative path.

    Layout: <h0>/<h1>/<hash><suffix>
    where h0=hash[:2], h1=hash[2:4].
    If hash cannot be parsed, return original key (plus suffix if provided).
    """

    def __init__(self, enable: bool = True):
        self.enable = enable

    def map(self, logical_key: str, suffix: str = "") -> str:
        if not self.enable:
            return f"{logical_key}{suffix}"
        hx = extract_hash_from_key(logical_key)
        if not hx or len(hx) < 4:
            return f"{logical_key}{suffix}"
        h0, h1 = hx[:2], hx[2:4]
        base = f"{hx}{suffix}"
        return os.path.join(h0, h1, base)


class MappedStorage(AbstractStorage):
    """Wrapper that rewrites file paths using a mapper before delegating.
    """

    def __init__(self, base: AbstractStorage, mapper: HashBucketPathMapper):
        self._base = base
        self._mapper = mapper

    def upload(self, file_path: str, data: bytes) -> bool:  # type: ignore[override]
        mapped = self._mapper.map(file_path)
        return self._base.upload(mapped, data)

    def download(self, file_path: str):  # type: ignore[override]
        mapped = self._mapper.map(file_path)
        return self._base.download(mapped)

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        mapped = self._mapper.map(file_path)
        return self._base.exists(mapped)

    def pack_kv_data(self, *args, **kwargs) -> bytes:  # type: ignore[override]
        return self._base.pack_kv_data(*args, **kwargs)

    def unpack_kv_data(self, data: bytes):  # type: ignore[override]
        return self._base.unpack_kv_data(data)

    def delete(self, file_path: str) -> bool:  # type: ignore[override]
        mapped = self._mapper.map(file_path)
        del_fn = getattr(self._base, "delete", None)
        if callable(del_fn):
            return bool(del_fn(mapped))
        return True
