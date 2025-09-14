# VLLM Adapter for Distributed KV Manager

This directory contains the adapter files needed to integrate the Distributed KV Manager with vLLM. To use this adapter, you need to place these files in the appropriate location within your vLLM installation.

## Installation

To integrate the Distributed KV Manager with your vLLM installation, follow these steps:

1. Locate your vLLM installation directory
2. Navigate to `vllm/distributed/kv_transfer/` within your vLLM installation
3. Copy all files from this directory to the vLLM `kv_transfer` directory:
   - `distributed_kv_connector.py`
   - `factory.py`

## File Placement

The files in this directory should be placed in the following locations within your vLLM installation:

```
vllm/
└── vllm/
    └── distributed/
        └── kv_transfer/
            ├── distributed_kv_connector.py
            └── factory.py
```

## Configuration

After placing the files, you need to configure vLLM to use the Distributed KV Manager. This is typically done by modifying the vLLM configuration to specify the KV connector.

### Registering the Connector

The `factory.py` file registers the `DistributedKVConnector` with the KV connector factory. Ensure that your vLLM configuration includes:

```python
# In your vLLM configuration
kv_transfer_config = {
    "kv_connector": "DistributedKVConnector",
    # Add other configuration options as needed
}
```

### Dependencies

Make sure that the `distributed_kv_manager` package is available in your Python environment. You can install it using:

```bash
pip install -e /path/to/distributed_kv_manager
```

## Usage

Once properly installed and configured, the Distributed KV Manager will automatically handle KV cache storage and retrieval operations during vLLM inference.

The adapter provides two main functionalities:
1. `recv_kv_caches_and_hidden_states`: Retrieves KV caches and hidden states from the distributed storage
2. `send_kv_caches_and_hidden_states`: Stores KV caches and hidden states to the distributed storage

## Troubleshooting

If you encounter issues:

1. Ensure all files are copied to the correct location
2. Verify that the `distributed_kv_manager` package is properly installed
3. Check that your vLLM version is compatible with this adapter
4. Confirm that your configuration correctly specifies the `DistributedKVConnector`

## Updating

When updating the Distributed KV Manager, you may need to update these adapter files as well. Always backup your existing files before updating.