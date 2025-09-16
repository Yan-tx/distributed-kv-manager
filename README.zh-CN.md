# 分布式KV管理器

一个为大语言模型推理场景设计的分布式键值缓存管理器，用于高效存储和检索KV（键值）缓存。该系统提供了高性能的缓存功能，支持Crail和本地文件系统等多种分布式存储后端。

## 功能特性

- **分布式KV缓存**：为LLM推理高效存储和检索KV缓存
- **多种存储后端**：支持Crail和本地文件系统存储
- **元数据管理**：使用ETCD进行强大的元数据管理，支持连接池和故障转移
- **缓存层**：三层元数据缓存，提升性能
- **异步操作**：基于线程池的非阻塞存储操作
- **容错性**：ETCD元数据的自动故障转移和复制
- **灵活配置**：轻松配置存储后端和缓存策略
- **自动清理**：基于可配置过期时间自动清理过期的KV缓存
- **元数据恢复**：能够从存储文件中恢复元数据
- **增强日志**：详细的日志记录便于调试和监控

## 架构

系统由以下几个关键组件构成：

1. **KV引擎**：处理存储和检索操作的核心引擎
2. **存储层**：抽象存储接口，支持Crail和本地存储实现
3. **元数据管理器**：基于ETCD的元数据管理，支持连接池
4. **元数据缓存**：三层缓存系统，提升元数据访问性能
5. **存储工厂**：工厂模式，用于创建适当的存储后端
6. **清理管理器**：自动清理过期的KV缓存

### 核心组件

#### KV引擎
协调KV缓存存储和检索操作的主引擎。它处理：
- 确定何时存储或检索KV缓存
- 管理异步存储操作
- 协调存储后端和元数据系统
- 在检索操作期间检查元数据过期

#### 存储后端
支持多种存储后端：
- **Crail存储**：高性能分布式存储系统
- **本地存储**：基于文件系统的本地存储，适用于开发/测试

#### 元数据管理
使用ETCD进行强大的元数据管理，具备：
- 多个ETCD节点的连接池
- 自动故障转移和复制
- 固定大小序列化的结构化元数据
- 元数据变更的监听能力
- 用于清理操作的元数据扫描
- 从存储文件中恢复元数据

#### 元数据缓存
三层元数据缓存系统：
- **Pool 1**：基于会话的元数据缓存
- **Pool 2**：基于层的元数据缓存
- **Pool 3**：最近访问的LRU缓存

#### 清理管理器
自动清理过期的KV缓存：
- 定期扫描元数据中的过期条目
- 可配置的清理间隔
- 删除过期条目的元数据和存储文件
- 访问时延长元数据生命周期（touch机制）

## 安装

```bash
# 克隆仓库
git clone https://github.com/Yan-tx/distributed-kv-manager.git
cd distributed_kv_manager

# 安装包
pip install -e .
```

## 配置

系统可以通过`kv_transfer_config`对象进行配置，包含以下选项：

```python
config.kv_transfer_config = SimpleNamespace(
    storage_type="crail",        # 或 "local"
    storage_dir="/kvcache",      # 存储的基础目录
    etcd_endpoints=["127.0.0.1:2379"],  # ETCD端点
    # SSD缓存配置
    enable_ssd_caching=False,    # 是否启用SSD缓存
    ssd_cache_dir="/tmp/ssd_cache",  # SSD缓存目录
    enable_prefetch=True,        # 是否启用预取
    # KV缓存自动淘汰配置
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
```

### 存储配置

- **Crail存储**：设置`storage_type="crail"`并配置`crail_dir`
- **本地存储**：设置`storage_type="local"`并配置`local_dir`

## 使用方法

### 基本使用

```python
from distributed_kv_manager import init_engine, destroy_engine
from distributed_kv_manager import should_store, store_kv, should_retrieve, retrieve_kv

# 初始化引擎
engine = init_engine(config)

# 检查是否应该存储KV缓存
store_status = should_store(model_input)

# 如需要则存储KV缓存
if store_status == StoreStatus.STORED:
    store_kv(model_config, parallel_config, transfer_config,
             model_executable, model_input, kv_caches, store_status)

# 检查是否可以检索KV缓存
retrieve_status = should_retrieve(model_input)

# 如可用则检索KV缓存
if retrieve_status == RetrieveStatus.HIT:
    hidden_state, bypass_model, new_input = retrieve_kv(
        model_executable, model_input, kv_caches, retrieve_status)

# 清理
destroy_engine()
```

### 自动清理

KV管理器会根据配置的过期时间自动清理过期的KV缓存。清理过程在后台线程中定期运行。您可以在配置中设置过期时间和清理间隔：

```python
config.kv_transfer_config = SimpleNamespace(
    # ... 其他配置选项
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
```

过期的KV缓存会自动从存储和元数据中删除，无需手动干预即可释放资源。

### 测试

运行测试套件以验证功能：

```bash
python test_kv_engine.py
```

## API参考

### 枚举

- `StoreStatus`：存储操作状态（STORED, SKIPPED）
- `RetrieveStatus`：检索操作状态（HIT, MISS）

### 函数

- `init_engine(config)`：使用给定配置初始化KV引擎
- `destroy_engine()`：清理并销毁引擎实例
- `should_store(model_input)`：确定是否应存储KV缓存
- `store_kv(...)`：使用给定参数存储KV缓存
- `should_retrieve(model_input)`：确定是否可检索KV缓存
- `retrieve_kv(...)`：如可用则检索KV缓存

## 开发

### 项目结构

```
distributed_kv_manager/
├── engine/              # 核心KV引擎实现
├── metadata/            # 元数据管理和缓存
├── storage/             # 存储后端实现
├── __init__.py          # 包初始化
└── tests/               # 测试文件
```

### 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试文件
python test_kv_engine.py

# 运行清理测试
python test_kv_cleanup.py
```

清理测试验证自动过期和清理功能。这些测试使用较短的过期时间和清理间隔来快速验证清理机制。

## 联系方式

如有问题或需要支持，请联系维护人员。