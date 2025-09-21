# Docker 部署说明

## 概述

本目录包含用于部署分布式KV管理器的Docker配置文件。我们提供了两个不同的Dockerfile以满足不同的部署需求：

- `Dockerfile`: 完整版本，包含Crail分布式存储支持
- `Dockerfile.no-crail`: 精简版本，仅支持本地存储

## Dockerfile 说明

### Dockerfile (完整版)
包含所有组件：
- Crail分布式存储系统
- Disni RDMA库
- SPDK存储驱动
- ETCD元数据管理
- vLLM推理引擎
- 分布式KV管理器

适用于生产环境和需要高性能分布式存储的场景。

### Dockerfile.no-crail (精简版)
仅包含核心组件：
- ETCD元数据管理
- vLLM推理引擎
- 分布式KV管理器（仅本地存储）

适用于开发测试环境或不需要分布式存储的场景。

## 构建镜像

### 构建完整版镜像
```bash
cd docker
docker build -t distributed-kv-manager:full .
```

### 构建精简版镜像
```bash
cd docker
docker build -f Dockerfile.no-crail -t distributed-kv-manager:lite .
```

## 使用说明

### 使用完整版
如果您需要使用Crail分布式存储功能：
1. 确保您的环境中可以安装并配置Crail集群
2. 构建完整版Docker镜像
3. 运行容器时需要挂载相关配置和数据卷

### 使用精简版
如果您只需要本地存储功能：
1. 构建精简版Docker镜像
2. 直接运行容器即可

## 注意事项

1. **vLLM适配器**: 如果您要使用带有Crail的Dockerfile，请不要添加vllm_adapter，因为文件夹中的vLLM已被修改。您只能通过创建和运行其docker来使用此版本。

2. **依赖关系**: 完整版镜像包含大量依赖组件，构建时间较长，请确保网络连接稳定。

3. **资源需求**: 完整版镜像较大，运行时需要足够的磁盘空间和内存。

4. **配置文件**: 运行容器时可能需要根据实际环境调整配置文件。

## 常见问题

### 构建失败
如果在构建过程中遇到网络问题，可以：
- 使用国内镜像源（Dockerfile中已配置）
- 手动下载依赖包并复制到构建上下文

### 运行时错误
如果容器运行时出现问题：
- 检查日志输出
- 验证配置文件是否正确
- 确认依赖服务（如ETCD）是否正常运行

## vLLM 快速启动与测试（容器内/宿主机均可）

以下为使用本项目连接器启动 vLLM（v0 接口）并做简单验证的命令示例：

### 启动 vLLM OpenAI 兼容服务（v0 接口）

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

### 启动本地 ETCD（如未启动）

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

### 基础请求测试

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

### 重要提醒（路径问题）

- 启动上述 `api_server` 时，请不要在 `~` 或 `~/vllm` 等与源码重名/冲突的目录下执行，以免因 Python 模块搜索路径冲突（例如本地目录名覆盖 `vllm` 包）导致导入失败或启动异常。
- 建议在独立的工作目录（如 `/workspace` 或任意非 `vllm` 命名的目录）中启动服务。
