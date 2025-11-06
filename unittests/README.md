# 单元测试与集成测试指南（Windows PowerShell）

该目录包含按模块划分的最小化单元测试，默认尽量规避外部依赖（未安装 torch 时相关用例会自动跳过；不要求本机已有 ETCD）：

- engine: 引擎初始化、should_store 和基本的 store/retrieve 路径（通过 Dummy 元数据管理器避免真实 ETCD 依赖）
- storage: LocalStorage 打包/上传/下载/解包与 exists/delete
- metadata: KVCleanupManager 在共享 hash 情境下的“安全删除”逻辑
- config: 配置加载与嵌套结构转换
- prefetch: PrefetchBuffer 状态机

如果需要跑“全量测试”（仓库根目录的 `tests/` + 本目录 `unittests/`），则需准备 PyTorch 与本地 ETCD。以下为详细步骤。

## 一、只跑纯单元测试（无外部依赖）

说明：仅运行 `unittests/`，其中张量相关用例在未安装 torch 时会自动跳过。

```pwsh
# 1) 在仓库根目录创建并激活虚拟环境（可选）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) 安装项目与测试依赖
pip install -e .
pip install pytest

# 3) 运行纯单元测试
python -m pytest unittests -q
```

等价的 unittest 原生命令：

```pwsh
python -m unittest discover -s unittests -p "test_*.py" -v
```

## 二、跑全量测试（tests/ + unittests/）

说明：`tests/` 目录中多数用例需要 ETCD 和 PyTorch。

1) 安装依赖

```pwsh
# 建议仍在虚拟环境内
pip install -e .
pip install pytest etcd3

# 安装 PyTorch（请按官方指引选择你机器的 CPU/CUDA 对应轮子）
# 官方文档：https://pytorch.org/get-started/locally/
# 示例（仅供参考，实际请以官方页面给出的命令为准）：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

2) 启动本地 ETCD（推荐 Docker）

```pwsh
docker run --name etcd --rm -p 2379:2379 -p 2380:2380 `
	quay.io/coreos/etcd:latest `
	/usr/local/bin/etcd `
	--name s1 `
	--data-dir /etcd-data `
	--listen-client-urls http://0.0.0.0:2379 `
	--advertise-client-urls http://127.0.0.1:2379 `
	--listen-peer-urls http://0.0.0.0:2380 `
	--initial-advertise-peer-urls http://127.0.0.1:2380 `
	--initial-cluster s1=http://127.0.0.1:2380 `
	--initial-cluster-token tkn `
	--initial-cluster-state new
```

3) 运行全量测试

```pwsh
python -m pytest tests unittests -q
```

## 三、常用变体

- 指定单文件（示例：引擎主流程）：

```pwsh
python -m pytest tests/test_kv_engine.py -q
```

- 只看覆盖率（需要安装 pytest-cov）：

```pwsh
pip install pytest-cov
python -m pytest tests unittests -q --cov=distributed_kv_manager --cov-report=term-missing
```

- 使用关键字过滤：

```pwsh
python -m pytest -k "config or prefetch" -q
```

## 四、排错与备注

- torch 未安装时，engine/storage 里使用张量的用例会自动跳过；可先运行纯单测保障基础行为。
- 未启动 ETCD 时，`tests/` 目录下涉及元数据与清理的用例会失败或阻塞；请确认 `127.0.0.1:2379` 可用。
- Windows PowerShell 中的 Docker 命令已使用反引号换行（`）。若使用 CMD，请改为单行或使用 ^ 续行；在 Git Bash/WSL 中用 `\` 续行。
- 测试可能在临时目录下生成数据（例如 `tmp_local_storage_*`），失败后如需手动清理，请删除对应目录。

## 五、后续扩展建议

1. 补充 CompositeStorage 分片文件名（`.L{split}.front/back`）与合并逻辑测试
2. 增加 MetadataCache 三层缓存命中率与过期行为测试
3. Prefetch 侧增加 IOAggregator 的速率控制与窗口聚合测试（通过替换内部 fetch_fn/on_ready 为计数器）
4. 引擎层添加“全量命中后 bypass 行为与 hidden_placeholder 形状校验”的端到端用例
5. 针对 HashBucket 路径映射做反向验证（与提取的 hash 一致）
