# Distributed KV Manager

A distributed key-value cache manager designed for efficient storage and retrieval of KV (Key-Value) caches in large language model inference scenarios. This system provides high-performance caching with support for distributed storage backends like Crail and local filesystems.

## Features

- **Distributed KV Caching**: Efficiently store and retrieve KV caches for LLM inference
- **Multiple Storage Backends**: Support for Crail and local filesystem storage
- **Metadata Management**: Robust metadata management using ETCD with connection pooling and failover
- **Caching Layer**: Three-tier metadata caching for improved performance
- **Asynchronous Operations**: Non-blocking storage operations with thread pool execution
- **Fault Tolerance**: Automatic failover and replication for ETCD metadata
- **Flexible Configuration**: Easy configuration of storage backends and cache policies
- **Automatic Cleanup**: Automatic cleanup of expired KV caches based on configurable expiration time
- **Metadata Recovery**: Ability to recover metadata from storage files
- **Enhanced Logging**: Detailed logging for debugging and monitoring

## Architecture

The system consists of several key components:

1. **KV Engine**: Core engine that handles storage and retrieval operations
2. **Storage Layer**: Abstract storage interface with implementations for Crail and local storage
3. **Metadata Manager**: ETCD-based metadata management with connection pooling
4. **Metadata Cache**: Three-tier caching system for improved metadata access performance
5. **Storage Factory**: Factory pattern for creating appropriate storage backends
6. **Cleanup Manager**: Automatic cleanup of expired KV caches

### Key Components

#### KV Engine
The main engine that orchestrates KV cache storage and retrieval operations. It handles:
- Determining when to store or retrieve KV caches
- Managing asynchronous storage operations
- Coordinating with storage backends and metadata systems
- Checking metadata expiration during retrieval operations

#### Storage Backends
Support for multiple storage backends:
- **Crail Storage**: High-performance distributed storage system
- **Local Storage**: Filesystem-based local storage for development/testing

#### Metadata Management
Robust metadata management using ETCD with:
- Connection pooling for multiple ETCD nodes
- Automatic failover and replication
- Structured metadata with fixed-size serialization
- Watch capabilities for metadata changes
- Metadata scanning for cleanup operations
- Metadata recovery from storage files

#### Metadata Caching
Three-tier caching system for metadata:
- **Pool 1**: Session-based metadata cache
- **Pool 2**: Layer-based metadata cache
- **Pool 3**: LRU cache for recent accesses

#### Cleanup Manager
Automatic cleanup of expired KV caches:
- Periodic scanning of metadata for expired entries
- Configurable cleanup interval
- Deletion of both metadata and storage files for expired entries
- Extension of metadata lifetime on access (touch mechanism)

## Installation

```bash
# Clone the repository
git clone https://github.com/Yan-tx/distributed-kv-manager.git
cd distributed_kv_manager

# Install the package
pip install -e .
```

### Local ETCD Setup for Development

For local development and testing, you can run a minimal ETCD instance with the following commands:

```bash
# Navigate to your etcd directory
cd ~/etcd

# Start ETCD in the background
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

This will start an ETCD server listening on port 2379, which is the default port expected by the KV manager.

## Configuration

The system can be configured in two ways:

### 1. Using Python Configuration Objects

The system can be configured through the `kv_transfer_config` object with the following options:

```python
config.kv_transfer_config = SimpleNamespace(
    storage_type="crail",        # or "local"
    storage_dir="/kvcache",      # Base directory for storage
    etcd_endpoints=["127.0.0.1:2379"],  # ETCD endpoints
    # SSD缓存配置
    enable_ssd_caching=False,    # 是否启用SSD缓存
    ssd_cache_dir="/tmp/ssd_cache",  # SSD缓存目录
    enable_prefetch=True,        # 是否启用预取
    # KV缓存自动淘汰配置
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600,       # 清理间隔时间（秒），默认1小时
    # 存储特定配置
    crail_dir="./crail_kvcache", # Crail存储目录
    local_dir="./local_kvcache"  # 本地存储目录
)
```

### 2. Using JSON Configuration File

Alternatively, you can create a `config.json` file in the project root directory:

```json
{
  "kv_transfer_config": {
    "storage_type": "crail",
    "storage_dir": "/kvcache",
    "etcd_endpoints": ["127.0.0.1:2379"],
    "enable_ssd_caching": false,
    "ssd_cache_dir": "/tmp/ssd_cache",
    "enable_prefetch": true,
    "kv_expire_time": 86400,
    "cleanup_interval": 3600,
    "crail_dir": "./crail_kvcache",
    "local_dir": "./local_kvcache"
  },
  "rank": 0,
  "local_rank": 0
}
```

Then initialize the engine without passing a config object:

```python
from distributed_kv_manager import init_engine

# Will automatically load from config.json
engine = init_engine()
```

### Storage Configuration

- **Crail Storage**: Set `storage_type="crail"` and configure `crail_dir`
- **Local Storage**: Set `storage_type="local"` and configure `local_dir`
- **SSD Caching**: Enable SSD caching by setting `enable_ssd_caching=True` and configure `ssd_cache_dir`

## Usage

### Basic Usage

```python
from distributed_kv_manager import init_engine, destroy_engine
from distributed_kv_manager import should_store, store_kv, should_retrieve, retrieve_kv

# Initialize the engine
engine = init_engine(config)

# Check if we should store KV cache
store_status = should_store(model_input)

# Store KV cache if needed
if store_status == StoreStatus.STORED:
    store_kv(model_config, parallel_config, transfer_config,
             model_executable, model_input, kv_caches, store_status)

# Check if we can retrieve KV cache
retrieve_status = should_retrieve(model_input)

# Retrieve KV cache if available
if retrieve_status == RetrieveStatus.HIT:
    hidden_state, bypass_model, new_input = retrieve_kv(
        model_executable, model_input, kv_caches, retrieve_status)

# Clean up
destroy_engine()
```

### Automatic Cleanup

The KV manager automatically cleans up expired KV caches based on the configured expiration time. The cleanup process runs periodically in a background thread. You can configure the expiration time and cleanup interval in the configuration:

```python
config.kv_transfer_config = SimpleNamespace(
    # ... other configuration options
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
```

Expired KV caches are automatically removed from both storage and metadata, freeing up resources without manual intervention.

### Testing

Run the test suite to verify functionality:

```bash
python test_kv_engine.py
```

## API Reference

### Enums

- `StoreStatus`: Status for storage operations (STORED, SKIPPED)
- `RetrieveStatus`: Status for retrieval operations (HIT, MISS)

### Functions

- `init_engine(config)`: Initialize the KV engine with the given configuration
- `destroy_engine()`: Clean up and destroy the engine instance
- `should_store(model_input)`: Determine if KV cache should be stored
- `store_kv(...)`: Store KV cache with the given parameters
- `should_retrieve(model_input)`: Determine if KV cache can be retrieved
- `retrieve_kv(...)`: Retrieve KV cache if available

## Development

### Project Structure

```
distributed_kv_manager/
├── engine/              # Core KV engine implementation
├── metadata/            # Metadata management and caching
├── storage/             # Storage backend implementations
├── __init__.py          # Package initialization
└── tests/               # Test files
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test files
python test_kv_engine.py

# Run cleanup tests
python test_kv_cleanup.py
```

The cleanup tests verify the automatic expiration and cleanup functionality. These tests use a shorter expiration time and cleanup interval to quickly validate the cleanup mechanism.

## vLLM Integration (Quick Start)

- Start a local ETCD first (if not already):

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

- Then start the vLLM OpenAI-compatible server with this connector (v0 API):

```bash
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
  --model /tmp/ckpt/Qwen --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'

VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
  --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'
```

- Simple requests to test:

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "写一首关于春天的诗。"}],
    "stream": true
  }'

curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "请用中文详细解释GAN的原理。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "请帮我写一首冬天的诗歌。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Important notes (path and model)

- Do not launch the `api_server` from `~`, `~/vllm`, or any directory whose
  name conflicts with installed packages (e.g., a folder named `vllm`). Such
  paths can shadow the real Python package and cause import/startup failures.
- Use an independent working directory instead (e.g., `/workspace`).
- Model path consistency: The `--model` path you pass when launching the
  server must exactly match the `"model"` field in your subsequent `curl`
  requests (e.g., both `/tmp/ckpt/Qwen3-0.6B`). A mismatch may cause
  load/routing failures.

## Contact

For questions or support, please contact the maintainers.

## Implementing a Custom Storage Backend

If you want to plug in your own storage (S3, OSS, NFS, etc.), implement the `AbstractStorage` interface and register it in the factory.

- Contract (see `distributed_kv_manager/storage/base.py`):
  - `upload(file_path: str, data: bytes) -> bool`
  - `download(file_path: str) -> Optional[bytes]`
  - `exists(file_path: str) -> bool`
  - `pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi) -> bytes`
  - `unpack_kv_data(data: bytes) -> (k_cache, v_cache, hidden)`
  - Optional helpers: `delete`, `list_files`, `extract_metadata_from_data`

- Packing/unpacking guidance
  - Follow `LocalStorage` as a reference. It packs CPU tensors via `torch.save` and returns CPU tensors from `unpack_kv_data`; the engine moves them to the right device later.
  - The engine may prepend embedded metadata bytes; it strips them before calling your `unpack_kv_data`, so you typically do not need to parse metadata.

- Minimal example

```python
# distributed_kv_manager/storage/my_storage.py
import io, os, torch, logging
from typing import Optional, Tuple
from .base import AbstractStorage

logger = logging.getLogger("MyStorage")

class MyStorage(AbstractStorage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def upload(self, file_path: str, data: bytes) -> bool:
        try:
            full = os.path.join(self.root_dir, file_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error("upload failed %s: %s", file_path, e)
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        full = os.path.join(self.root_dir, file_path)
        if not os.path.exists(full):
            return None
        with open(full, "rb") as f:
            return f.read()

    def exists(self, file_path: str) -> bool:
        return os.path.exists(os.path.join(self.root_dir, file_path))

    def pack_kv_data(self, k_cache, v_cache, hidden, input_tokens, roi) -> bytes:
        buf = io.BytesIO()
        torch.save({
            "k_cache": k_cache.cpu(),
            "v_cache": v_cache.cpu(),
            "hidden": None if hidden is None else hidden.cpu(),
            "input_tokens": input_tokens.cpu(),
            "roi": roi.cpu(),
        }, buf)
        return buf.getvalue()

    def unpack_kv_data(self, data: bytes):
        obj = torch.load(io.BytesIO(data), map_location="cpu")
        return obj["k_cache"], obj["v_cache"], obj.get("hidden", None)
```

- Register in the factory (`distributed_kv_manager/storage/factory.py`):

```python
from .my_storage import MyStorage  # add import

class StorageFactory:
    @staticmethod
    def create_storage(config):
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        # ...
        if storage_type == "my_storage":
            root = getattr(config.kv_transfer_config, "my_dir", "/tmp/kvcache_my")
            base_storage = MyStorage(root)
        # wrap with SSD cache if enable_ssd_caching=True
```

- Configure it (JSON or Python):

```json
{
  "kv_transfer_config": {
    "storage_type": "my_storage",
    "my_dir": "/data/kvcache",
    "enable_ssd_caching": false
  }
}
```

Tips
- Ensure `file_path` is a relative key (factory/backends handle the base dir).
- Keep `pack/unpack` stable across versions; shapes and dtypes must round-trip.
- If you enable the SSD wrapper, the same `file_path` must work for upload/download/exists.
