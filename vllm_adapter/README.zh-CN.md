# VLLM适配器 for 分布式KV管理器

此目录包含将分布式KV管理器与vLLM集成所需的适配器文件。要使用此适配器，您需要将这些文件放置在vLLM安装中的适当位置。

## 安装

要将分布式KV管理器与您的vLLM安装集成，请按照以下步骤操作：

1. 定位您的vLLM安装目录
2. 导航到vLLM安装中的`vllm/distributed/kv_transfer/`目录
3. 将此目录中的所有文件复制到vLLM的`kv_transfer`目录：
   - `distributed_kv_connector.py`
   - `factory.py`

## 文件放置

此目录中的文件应放置在vLLM安装中的以下位置：

```
vllm/
└── vllm/
    └── distributed/
        └── kv_transfer/
            ├── distributed_kv_connector.py
            └── factory.py
```

## 配置

放置文件后，您需要配置vLLM以使用分布式KV管理器。通常通过修改vLLM配置来指定KV连接器完成此操作。

### 注册连接器

`factory.py`文件将`DistributedKVConnector`注册到KV连接器工厂。确保您的vLLM配置包含：

```python
# 在您的vLLM配置中
kv_transfer_config = {
    "kv_connector": "DistributedKVConnector",
    # 根据需要添加其他配置选项
}
```

### 依赖项

确保`distributed_kv_manager`包在您的Python环境中可用。您可以使用以下命令安装：

```bash
pip install -e /path/to/distributed_kv_manager
```

## 使用方法

正确安装和配置后，分布式KV管理器将在vLLM推理期间自动处理KV缓存的存储和检索操作。

适配器提供两个主要功能：
1. `recv_kv_caches_and_hidden_states`：从分布式存储中检索KV缓存和隐藏状态
2. `send_kv_caches_and_hidden_states`：将KV缓存和隐藏状态存储到分布式存储中

## 故障排除

如果遇到问题：

1. 确保所有文件都已复制到正确位置
2. 验证`distributed_kv_manager`包是否已正确安装
3. 检查您的vLLM版本是否与此适配器兼容
4. 确认您的配置正确指定了`DistributedKVConnector`

## 更新

在更新分布式KV管理器时，您可能也需要更新这些适配器文件。更新前请务必备份现有文件。