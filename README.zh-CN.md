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

### 本地ETCD设置用于开发

为了本地开发和测试，您可以使用以下命令运行一个最小化的ETCD实例：

```bash
# 导航到您的etcd目录
cd ~/etcd

# 在后台启动ETCD
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

这将启动一个监听2379端口的ETCD服务器，这是KV管理器默认期望的端口。

## 配置

系统可以通过两种方式进行配置：

### 1. 使用Python配置对象

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
    cleanup_interval=3600,       # 清理间隔时间（秒），默认1小时
    # 存储特定配置
    crail_dir="./crail_kvcache", # Crail存储目录
    local_dir="./local_kvcache"  # 本地存储目录
)
```

### 2. 使用JSON配置文件

或者，您可以在项目根目录创建一个`config.json`文件：

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

然后在不传递配置对象的情况下初始化引擎：

```python
from distributed_kv_manager import init_engine

# 将自动从config.json加载配置
engine = init_engine()
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

## vLLM 快速启动与测试

- 先启动本地 ETCD（如未启动）：

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

- 然后使用本连接器启动 vLLM 的 OpenAI 兼容服务（v0 接口）：

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

- 基础请求测试：

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

### 重要提醒（路径与模型）

- 启动 `api_server` 时，请不要在 `~` 或 `~/vllm` 等与源码包同名/冲突的目录下执行，
  否则可能因 Python 模块搜索路径冲突（本地目录遮蔽 `vllm` 包）导致导入/启动失败。
- 建议在独立的工作目录（例如 `/workspace`）中启动服务。
- 模型路径一致性：启动 vLLM 时使用的 `--model` 路径必须与后续 `curl` 请求体中的 `"model"` 字段完全一致
  （例如都为 `/tmp/ckpt/Qwen3-0.6B`）。不一致可能导致加载/路由失败。

## 联系方式

如有问题或需要支持，请联系维护人员。

## 自定义存储后端实现

如果你希望接入自定义存储（S3/OSS/NFS/自研分布式存储等），可实现 `AbstractStorage` 并在工厂中注册。

- 接口约定（见 `distributed_kv_manager/storage/base.py`）：
  - `upload(file_path: str, data: bytes) -> bool`
  - `download(file_path: str) -> Optional[bytes]`
  - `exists(file_path: str) -> bool`
  - `pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi) -> bytes`
  - `unpack_kv_data(data: bytes) -> (k_cache, v_cache, hidden)`
  - 可选：`delete`、`list_files`、`extract_metadata_from_data`

- 打包/解包建议
  - 可参考 `LocalStorage`：用 `torch.save` 将 CPU 张量写入字节流；`unpack_kv_data` 返回 CPU 张量，设备迁移由引擎负责。
  - 引擎可能会在写入前嵌入元数据（头部+定长结构），读取时会先剥离再调用你的 `unpack_kv_data`，通常无需自行解析元数据。

- 最小示例

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

- 在工厂中注册（`distributed_kv_manager/storage/factory.py`）：

```python
from .my_storage import MyStorage  # 新增导入

class StorageFactory:
    @staticmethod
    def create_storage(config):
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        # ...
        if storage_type == "my_storage":
            root = getattr(config.kv_transfer_config, "my_dir", "/tmp/kvcache_my")
            base_storage = MyStorage(root)
        # 如需 SSD 包裹，在 enable_ssd_caching=True 时会被 CachingStorage 包裹
```

- 配置示例（JSON 或 Python 均可）：

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
- 确保 `file_path` 为相对键（基目录由后端/工厂拼接）。
- 保证 `pack/unpack` 的兼容性：形状与 dtype 必须能无损还原。
- 启用 SSD 包裹时，`upload/download/exists` 要保持对同一 `file_path` 的一致语义。
