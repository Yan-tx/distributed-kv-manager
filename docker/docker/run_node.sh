#!/bin/bash

set -e

if pgrep -f "org.apache.crail.namenode.NameNode" > /dev/null; then
  echo "NameNode 已在运行，启动 DataNode..."
  $CRAIL_HOME/bin/crail datanode -t org.apache.crail.storage.nvmf.NvmfStorageTier -c 1
  exit 0
fi

echo "=== 设置 Transparent HugePage ==="
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

echo "=== 挂载 hugetlbfs (如果没挂载) ==="
if ! mount | grep -q "/mnt/hugepages/data"; then
  sudo mkdir -p /mnt/hugepages/data
  sudo mount -t hugetlbfs nodev /mnt/hugepages/data
fi
if ! mount | grep -q "/mnt/hugepages/cache"; then
  sudo mkdir -p /mnt/hugepages/cache
  sudo mount -t hugetlbfs nodev /mnt/hugepages/cache
fi
chmod -R 777 /mnt/hugepages/data /mnt/hugepages/cache

echo "=== 启动 SPDK nvmf_tgt ==="
cd /root/spdk
sudo ./build/bin/nvmf_tgt &
NVMF_PID=$!

sleep 3

echo "=== 创建 bdev 控制器 ==="
# 示例：请根据实际 PCI 地址和 bdev 名称修改
sudo ./scripts/rpc.py bdev_nvme_attach_controller -b bdev0 -a 0000:5f:00.0 -t PCIe

echo "=== 创建 RDMA 传输通道 ==="
sudo ./scripts/rpc.py nvmf_create_transport -t RDMA -u 8192 -i 131072 -c 8192

echo "=== 创建 NVMf 子系统 ==="
SUBSYSTEM_NQN="nqn.2017-06.io.crail:cnode1"
SUBSYSTEM_SN="SPDK000000000001"
sudo ./scripts/rpc.py nvmf_create_subsystem $SUBSYSTEM_NQN -a -s $SUBSYSTEM_SN

echo "=== 添加 bdev 到子系统 ==="
sudo ./scripts/rpc.py nvmf_subsystem_add_ns $SUBSYSTEM_NQN bdev0n1

echo "=== 添加监听器 ==="
# 请修改为实际的 IB 网卡地址和监听端口
IB_ADDR="192.168.100.10"
IB_PORT="4420"
sudo ./scripts/rpc.py nvmf_subsystem_add_listener $SUBSYSTEM_NQN -t RDMA -a $IB_ADDR -s $IB_PORT

echo "=== 启动 Crail ==="
# 你可以传参数：namenode、datanode 

MODE=${1:-datanode}

if [ "$MODE" == "namenode" ]; then
  $CRAIL_HOME/bin/crail namenode
elif [ "$MODE" == "datanode" ]; then
  $CRAIL_HOME/bin/crail datanode -t org.apache.crail.storage.nvmf.NvmfStorageTier -c 1
else
  echo "Usage: $0 [namenode|datanode]"
  exit 1
fi

echo "=== 脚本执行完毕 ==="
