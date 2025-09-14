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
