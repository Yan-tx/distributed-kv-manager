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

## Architecture

The system consists of several key components:

1. **KV Engine**: Core engine that handles storage and retrieval operations
2. **Storage Layer**: Abstract storage interface with implementations for Crail and local storage
3. **Metadata Manager**: ETCD-based metadata management with connection pooling
4. **Metadata Cache**: Three-tier caching system for improved metadata access performance
5. **Storage Factory**: Factory pattern for creating appropriate storage backends

### Key Components

#### KV Engine
The main engine that orchestrates KV cache storage and retrieval operations. It handles:
- Determining when to store or retrieve KV caches
- Managing asynchronous storage operations
- Coordinating with storage backends and metadata systems

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

#### Metadata Caching
Three-tier caching system for metadata:
- **Pool 1**: Session-based metadata cache
- **Pool 2**: Layer-based metadata cache
- **Pool 3**: LRU cache for recent accesses

## Installation

```bash
# Clone the repository
git clone https://github.com/Yan-tx/distributed-kv-manager.git
cd distributed_kv_manager

# Install the package
pip install -e .
```

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
```

## Contact

For questions or support, please contact the maintainers.