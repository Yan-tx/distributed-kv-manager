import os
import io
import subprocess
import torch
import logging
from typing import Optional, Tuple
from .base import AbstractStorage

logger = logging.getLogger("CrailStorage")

class CrailStorage(AbstractStorage):
    """封装Crail文件操作，实现AbstractStorage接口"""
    
    def __init__(self, crail_dir: str):
        self.crail_dir = crail_dir
        os.makedirs(self.crail_dir, exist_ok=True)
        
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到Crail"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "upload-stream", file_path
        ]
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate(input=data, timeout=30)
            if process.returncode != 0:
                logger.error(f"Crail upload failed: {stderr.decode()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Crail upload timed out")
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        """从Crail下载数据"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "download-stream", file_path
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunks = []
            while True:
                chunk = process.stdout.read(16 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            stderr = process.stderr.read()
            process.wait()
            if process.returncode != 0:
                logger.error(f"Crail download failed: {stderr.decode()}")
                return None
            return b"".join(chunks)
        except Exception as e:
            logger.error(f"Crail download error: {e}")
            return None

    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "exists", file_path
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=10)
            return process.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def pack_kv_data(self, k_cache: torch.Tensor, v_cache: torch.Tensor, 
                    hidden: Optional[torch.Tensor], input_tokens: torch.Tensor, 
                    roi: torch.Tensor) -> bytes:
        """打包KV数据为字节流"""
        data = {
            "k_cache": k_cache.cpu(),
            "v_cache": v_cache.cpu(),
            "hidden": hidden.cpu() if hidden is not None else None,
            "input_tokens": input_tokens.cpu(),
            "roi": roi.cpu()
        }
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据"""
        try:
            buffer = io.BytesIO(data)
            loaded = torch.load(buffer, map_location="cpu")
            return loaded["k_cache"], loaded["v_cache"], loaded.get("hidden", None)
        except Exception as e:
            logger.error(f"Failed to unpack KV data: {e}")
            return None, None, None