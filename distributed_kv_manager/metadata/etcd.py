import struct
import time
import threading
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import etcd3
from etcd3 import Etcd3Client


# 在文件开头添加常量定义
METADATA_HEADER = b"KVMD"
METADATA_VERSION = 1
HEADER_SIZE = 8  # METADATA_HEADER (4 bytes) + METADATA_VERSION (4 bytes)
METADATA_SIZE = 320  # KVMetadata size in bytes

@dataclass
class KVMetadata:
    session_id: bytes       # 16B
    layer_id: int           # 4B
    token_idx: str          # "1024-2048", 存储为16B
    file_path: str          # 128B
    file_size: int          # 8B
    create_time: int        # 8B (epoch秒)
    last_access: int        # 8B
    expire_time: int        # 8B (过期时间，单位秒，0表示永不过期)
    replica_locations: List[bytes]  # 3 * 16B = 48B
    status: int             # 1B
    schema_version: int     # 2B
    ext_flags: int          # 4B
    ext_data: bytes         # 64B
    ext_data_len: int       # 4B
    reserved: bytes = b"\x00" * 1  # 1B 对齐 (调整了对齐字节以适应新字段)

    def pack(self) -> bytes:
        """序列化为320字节"""
        return struct.pack(
            "<16sI16s128sQQQQ48sBHI64sI1s",
            self.session_id.ljust(16, b"\x00"),
            self.layer_id,
            self.token_idx.encode("utf-8").ljust(16, b"\x00"),
            self.file_path.encode("utf-8").ljust(128, b"\x00"),
            self.file_size,
            self.create_time,
            self.last_access,
            self.expire_time,
            b"".join(self.replica_locations).ljust(48, b"\x00"),
            self.status,
            self.schema_version,
            self.ext_flags,
            self.ext_data.ljust(64, b"\x00"),
            self.ext_data_len,
            self.reserved,
        )

    @staticmethod
    def unpack(data: bytes) -> "KVMetadata":
        fields = struct.unpack("<16sI16s128sQQQQ48sBHI64sI1s", data)
        return KVMetadata(
            session_id=fields[0].rstrip(b"\x00"),
            layer_id=fields[1],
            token_idx=fields[2].decode("utf-8").rstrip("\x00"),
            file_path=fields[3].decode("utf-8").rstrip("\x00"),
            file_size=fields[4],
            create_time=fields[5],
            last_access=fields[6],
            expire_time=fields[7],
            replica_locations=[
                fields[8][i:i+16] for i in range(0, 48, 16)
            ],
            status=fields[9],
            schema_version=fields[10],
            ext_flags=fields[11],
            ext_data=fields[12].rstrip(b"\x00"),
            ext_data_len=fields[13],
            reserved=fields[14],
        )
        
    def to_dict(self) -> Dict:
        """将元数据转换为字典格式，用于存储到文件中"""
        return {
            "session_id": self.session_id.hex(),
            "layer_id": self.layer_id,
            "token_idx": self.token_idx,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "create_time": self.create_time,
            "last_access": self.last_access,
            "expire_time": self.expire_time,
            "replica_locations": [loc.hex() for loc in self.replica_locations],
            "status": self.status,
            "schema_version": self.schema_version,
            "ext_flags": self.ext_flags,
            "ext_data": self.ext_data.hex(),
            "ext_data_len": self.ext_data_len,
        }
        
    @staticmethod
    def from_dict(data: Dict) -> "KVMetadata":
        """从字典格式恢复元数据"""
        return KVMetadata(
            session_id=bytes.fromhex(data["session_id"]),
            layer_id=data["layer_id"],
            token_idx=data["token_idx"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            create_time=data["create_time"],
            last_access=data["last_access"],
            expire_time=data.get("expire_time", 0),  # 默认为0表示永不过期
            replica_locations=[bytes.fromhex(loc) for loc in data["replica_locations"]],
            status=data["status"],
            schema_version=data["schema_version"],
            ext_flags=data["ext_flags"],
            ext_data=bytes.fromhex(data["ext_data"]),
            ext_data_len=data["ext_data_len"],
        )
        
    def pack_with_embedding(self) -> bytes:
        """将元数据打包成嵌入格式，用于存储到数据文件中"""
        header = struct.pack("<4sI", METADATA_HEADER, METADATA_VERSION)
        metadata_bytes = self.pack()
        return header + metadata_bytes
        
    @staticmethod
    def unpack_from_file(data: bytes) -> Optional["KVMetadata"]:
        """从文件中解包元数据"""
        if len(data) < HEADER_SIZE:
            return None
            
        header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
        if header != METADATA_HEADER or version != METADATA_VERSION:
            return None
            
        if len(data) < HEADER_SIZE + 320:  # HEADER_SIZE + KVMetadata size
            return None
            
        return KVMetadata.unpack(data[HEADER_SIZE:HEADER_SIZE + 320])

    def is_expired(self) -> bool:
        """检查元数据是否已过期"""
        # 如果expire_time为0，表示永不过期
        if self.expire_time == 0:
            return False
            
        current_time = int(time.time())
        # 如果当前时间减去最后访问时间大于等于过期时间，则已过期
        return (current_time - self.last_access) >= self.expire_time


class EtcdConnectionPool:
    """ETCD连接池，管理多个etcd节点的连接"""
    
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.clients = {}
        self.lock = threading.RLock()
        self._init_connections()
    
    def _init_connections(self):
        """初始化所有etcd节点的连接"""
        for ep in self.endpoints:
            try:
                host, port = ep.split(":")
                client = etcd3.client(host=host, port=int(port))
                # 测试连通性
                client.status()
                self.clients[ep] = client
                print(f"[EtcdConnectionPool] Connected to etcd {ep}")
            except Exception as e:
                print(f"[EtcdConnectionPool] Failed to connect {ep}: {e}")
                self.clients[ep] = None
        
        if not any(self.clients.values()):
            raise RuntimeError("No available etcd endpoints!")
    
    def get_connection(self, preferred_endpoint: Optional[str] = None) -> Etcd3Client:
        """获取一个可用的etcd连接"""
        with self.lock:
            # 如果指定了优先节点且该节点可用，则返回
            if preferred_endpoint and preferred_endpoint in self.clients:
                client = self.clients[preferred_endpoint]
                if client and self._check_connection(client):
                    return client
            
            # 否则返回第一个可用的连接
            for endpoint, client in self.clients.items():
                if client and self._check_connection(client):
                    return client
            
            # 如果没有可用连接，尝试重新连接
            self._reconnect_all()
            for endpoint, client in self.clients.items():
                if client and self._check_connection(client):
                    return client
            
            raise RuntimeError("No available etcd connections!")
    
    def get_all_connections(self) -> list[Etcd3Client]:
        """获取所有可用的etcd连接"""
        with self.lock:
            available_clients = []
            for client in self.clients.values():
                if client and self._check_connection(client):
                    available_clients.append(client)
            return available_clients
    
    def _check_connection(self, client: Etcd3Client) -> bool:
        """检查连接是否有效"""
        try:
            client.status()
            return True
        except:
            return False
    
    def _reconnect_all(self):
        """重新连接所有不可用的节点"""
        for endpoint in self.clients:
            if not self._check_connection(self.clients[endpoint]):
                try:
                    host, port = endpoint.split(":")
                    client = etcd3.client(host=host, port=int(port))
                    client.status()
                    self.clients[endpoint] = client
                    print(f"[EtcdConnectionPool] Reconnected to etcd {endpoint}")
                except Exception as e:
                    print(f"[EtcdConnectionPool] Failed to reconnect {endpoint}: {e}")
                    self.clients[endpoint] = None


class KVMetadataManager:
    def __init__(self, endpoints: list[str], prefix: str = "/kvmeta", max_workers: int = 5):
        """
        endpoints: ["ip1:2379", "ip2:2379", "ip3:2379"]
        使用连接池管理多个etcd节点连接，提供故障转移能力
        max_workers: 用于多线程复制的最大线程数
        """
        self.endpoints = endpoints
        self.prefix = prefix
        self.connection_pool = EtcdConnectionPool(endpoints)
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
    def _execute_with_failover(self, operation, *args, **kwargs):
        """执行操作，如果失败则尝试故障转移"""
        last_exception = None
        
        # 尝试所有可用节点
        for client in self.connection_pool.get_all_connections():
            try:
                return operation(client, *args, **kwargs)
            except Exception as e:
                last_exception = e
                print(f"Operation failed on {client._url}: {e}")
                continue
        
        # 如果所有节点都失败，抛出异常
        raise last_exception or RuntimeError("Operation failed on all etcd nodes")
        
    def put_metadata(self, key: str, meta: KVMetadata, replicate: bool = True):
        """存储元数据，可选择是否多线程复制到所有节点"""
        if not replicate:
            # 单节点写入
            def _put(client):
                client.put(f"{self.prefix}/{key}", meta.pack())
            self._execute_with_failover(_put)
        else:
            # 多节点并行写入
            self._replicate_operation("put", key, meta)
            
    def get_metadata(self, key: str) -> Optional[KVMetadata]:
        """获取元数据，自动故障转移"""
        def _get(client):
            value, _ = client.get(f"{self.prefix}/{key}")
            return value
        
        value = self._execute_with_failover(_get)
        if value is None:
            return None
        return KVMetadata.unpack(value)
        
    def update_access_time(self, key: str, replicate: bool = True):
        """更新访问时间，可选择是否多线程复制到所有节点"""
        meta = self.get_metadata(key)
        if meta:
            meta.last_access = int(time.time())
            self.put_metadata(key, meta, replicate)
            
    def delete_metadata(self, key: str, replicate: bool = True):
        """删除元数据，可选择是否多线程复制到所有节点"""
        if not replicate:
            # 单节点删除
            def _delete(client):
                client.delete(f"{self.prefix}/{key}")
            self._execute_with_failover(_delete)
        else:
            # 多节点并行删除
            self._replicate_operation("delete", key)
            
    def _replicate_operation(self, operation: str, key: str, meta: Optional[KVMetadata] = None):
        """多线程复制操作到所有etcd节点"""
        clients = self.connection_pool.get_all_connections()
        futures = []
        
        for client in clients:
            if operation == "put":
                futures.append(
                    self.thread_pool.submit(
                        self._safe_put, client, f"{self.prefix}/{key}", meta.pack()
                    )
                )
            elif operation == "delete":
                futures.append(
                    self.thread_pool.submit(
                        self._safe_delete, client, f"{self.prefix}/{key}"
                    )
                )
        
        # 等待所有操作完成，记录失败的操作
        failed_operations = []
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
            except Exception as e:
                failed_operations.append((clients[i]._url, e))
        
        # 如果有失败的操作，记录日志或采取其他措施
        if failed_operations:
            print(f"Failed to replicate operation to {len(failed_operations)} nodes:")
            for url, error in failed_operations:
                print(f"  {url}: {error}")
                
    def _safe_put(self, client, key, value):
        try:
            client.put(key, value)
        except Exception as e:
            print(f"Failed to put {key} to {client._url}: {e}")
            raise
            
    def _safe_delete(self, client, key):
        try:
            client.delete(key)
        except Exception as e:
            print(f"Failed to delete {key} from {client._url}: {e}")
            raise
            
    def watch_metadata(self, key: str, callback):
        """订阅某个key的变更事件"""
        # 使用主节点进行watch
        client = self.connection_pool.get_connection()
        events_iterator, cancel = client.watch(f"{self.prefix}/{key}")
        for event in events_iterator:
            callback(event)
            
    def recover_metadata_from_storage(self, storage_path: str) -> Dict[str, KVMetadata]:
        """
        从存储中扫描并提取嵌入的元数据
        返回: {file_path: KVMetadata}
        """
        recovered_metadata = {}
        
        # 遍历存储路径下的所有文件
        for root, _, files in os.walk(storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, storage_path)
                
                # 尝试从文件中提取嵌入的元数据
                metadata = self._extract_embedded_metadata_from_file(file_path)
                if metadata:
                    # 更新文件路径为相对路径
                    metadata.file_path = relative_path
                    recovered_metadata[relative_path] = metadata
                    
        return recovered_metadata
        
    def _extract_embedded_metadata_from_file(self, file_path: str) -> Optional[KVMetadata]:
        """
        从文件中提取嵌入的元数据
        """
        try:
            # 读取文件的前几个字节，检查是否有元数据头部
            with open(file_path, 'rb') as f:
                # 读取足够多的字节以包含头部和元数据
                data = f.read(HEADER_SIZE + METADATA_SIZE)
                
            # 尝试解包元数据
            metadata = KVMetadata.unpack_from_file(data)
            if metadata:
                return metadata
                
            return None
        except Exception as e:
            print(f"Failed to extract embedded metadata from {file_path}: {e}")
            return None
            
    def write_metadata_to_etcd(self, metadata_dict: Dict[str, KVMetadata]):
        """
        将元数据写入etcd
        """
        for file_path, metadata in metadata_dict.items():
            try:
                # 使用文件路径作为key
                key = file_path
                self.put_metadata(key, metadata)
                print(f"Successfully wrote metadata for {file_path} to etcd")
            except Exception as e:
                print(f"Failed to write metadata for {file_path} to etcd: {e}")