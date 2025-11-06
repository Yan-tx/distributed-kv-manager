import io
import struct
import torch
from typing import Optional, Tuple

from .base import AbstractStorage
from ..metadata.etcd import HEADER_SIZE, METADATA_HEADER, METADATA_VERSION, METADATA_SIZE


class CompositeStorage(AbstractStorage):
    """
    将同一逻辑文件按“层”切分为两段分别存入本地与远端存储：
    - 逻辑名: file_path
    - 物理名: file_path + ".front"（前段，存入 local_tier）
             file_path + ".back"  （后段，存入 remote_tier）

    读取时自动合并两段为一个 payload bytes，接口对引擎保持透明。
    不改持久化元数据（ETCD 仍记录逻辑名）。
    """

    def __init__(
        self,
        local_tier: AbstractStorage,
        remote_tier: AbstractStorage,
        layer_split_front: Optional[int] = None,
        layer_split_ratio: float = 0.5,
    ):
        self.local_tier = local_tier
        self.remote_tier = remote_tier
        self.layer_split_front = layer_split_front
        self.layer_split_ratio = max(0.0, min(1.0, float(layer_split_ratio)))

    # ---- helpers ----
    def _strip_metadata_header(self, data: bytes) -> bytes:
        if len(data) >= HEADER_SIZE + METADATA_SIZE:
            try:
                header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
                if header == METADATA_HEADER and version == METADATA_VERSION:
                    return data[HEADER_SIZE + METADATA_SIZE :]
            except Exception:
                pass
        return data

    def _choose_split(self, num_layers: int) -> int:
        if num_layers <= 0:
            return 0
        if self.layer_split_front is not None:
            return max(0, min(num_layers, int(self.layer_split_front)))
        # ratio-based
        return max(0, min(num_layers, int(round(num_layers * self.layer_split_ratio))))

    # ---- AbstractStorage impl ----
    def upload(self, file_path: str, data: bytes) -> bool:
        # 去掉嵌入的元数据头，解析 torch.save 字典
        payload = self._strip_metadata_header(data)
        try:
            obj = torch.load(io.BytesIO(payload), map_location="cpu")
        except Exception:
            # 兜底：直接作为整块上传到远端
            return self.remote_tier.upload(file_path, data)

        k = obj.get("k_cache")
        v = obj.get("v_cache")
        if k is None or v is None or k.dim() == 0:
            # 内容不符合预期，整块走远端
            return self.remote_tier.upload(file_path, data)

        num_layers = int(k.shape[0])
        split_n = self._choose_split(num_layers)

        # 如果 split=0 或 =num_layers，则不切分，分别落在远端或本地
        if split_n <= 0:
            # 全量后段：远端
            return self.remote_tier.upload(file_path + ".back", payload)
        if split_n >= num_layers:
            # 全量前段：本地
            return self.local_tier.upload(file_path + ".front", payload)

        # 构建前后段 payload
        def build_part(k_part, v_part):
            part = {
                "k_cache": k_part.cpu(),
                "v_cache": v_part.cpu(),
            }
            # 保留其它字段（input_tokens/roi/slot_mapping/payload_meta）
            for kk in ("input_tokens", "roi", "slot_mapping", "payload_meta"):
                if kk in obj:
                    part[kk] = obj[kk]
            return part

        front = build_part(k[:split_n], v[:split_n])
        back = build_part(k[split_n:], v[split_n:])

        buf_front, buf_back = io.BytesIO(), io.BytesIO()
        torch.save(front, buf_front)
        torch.save(back, buf_back)

        ok1 = self.local_tier.upload(file_path + ".front", buf_front.getvalue())
        ok2 = self.remote_tier.upload(file_path + ".back", buf_back.getvalue())
        return ok1 and ok2

    def download(self, file_path: str) -> Optional[bytes]:
        # 优先尝试读两段；允许其中一段缺失
        front = self.local_tier.download(file_path + ".front")
        back = self.remote_tier.download(file_path + ".back")
        if front is None and back is None:
            return None

        def load_or_empty(chunk: Optional[bytes]):
            if chunk is None:
                return None
            try:
                return torch.load(io.BytesIO(chunk), map_location="cpu")
            except Exception:
                return None

        fobj = load_or_empty(front)
        bobj = load_or_empty(back)

        # 如果只有单段，直接返回原 chunk
        if fobj is None and bobj is not None:
            return back
        if bobj is None and fobj is not None:
            return front

        # 合并
        if not isinstance(fobj, dict) or not isinstance(bobj, dict):
            return back or front
        fk, fv = fobj.get("k_cache"), fobj.get("v_cache")
        bk, bv = bobj.get("k_cache"), bobj.get("v_cache")
        if fk is None or fv is None or bk is None or bv is None:
            # 不合规，优先返回 back
            return back or front

        merged = {
            "k_cache": torch.cat([fk, bk], dim=0),
            "v_cache": torch.cat([fv, bv], dim=0),
        }
        # 透传其它字段（以 front 为主）
        for kk in ("input_tokens", "roi", "slot_mapping", "payload_meta"):
            if fobj and kk in fobj:
                merged[kk] = fobj[kk]
            elif bobj and kk in bobj:
                merged[kk] = bobj[kk]

        buf = io.BytesIO()
        torch.save(merged, buf)
        return buf.getvalue()

    def exists(self, file_path: str) -> bool:
        return (
            self.local_tier.exists(file_path + ".front")
            or self.remote_tier.exists(file_path + ".back")
        )

    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        # 使用通用 torch.save 格式
        buf = io.BytesIO()
        torch.save({
            "k_cache": k_cache.cpu(),
            "v_cache": v_cache.cpu(),
            "input_tokens": input_tokens.cpu(),
            "roi": roi.cpu(),
        }, buf)
        return buf.getvalue()

    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            obj = torch.load(io.BytesIO(data), map_location="cpu")
            return obj.get("k_cache"), obj.get("v_cache")
        except Exception:
            return None, None

    # 可选：删除两个物理段
    def delete(self, file_path: str) -> bool:
        ok1 = True
        ok2 = True
        del_local = getattr(self.local_tier, "delete", None)
        del_remote = getattr(self.remote_tier, "delete", None)
        if callable(del_local):
            ok1 = bool(del_local(file_path + ".front"))
        if callable(del_remote):
            ok2 = bool(del_remote(file_path + ".back"))
        return ok1 and ok2
