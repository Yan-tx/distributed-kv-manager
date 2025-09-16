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
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
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

## Contact

For questions or support, please contact the maintainers.