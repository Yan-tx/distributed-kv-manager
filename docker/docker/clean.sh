#!/bin/bash

set -e

echo "=== 停止 SPDK nvmf_tgt ==="
NVMF_PID=$(pgrep -f nvmf_tgt || true)
if [ -n "$NVMF_PID" ]; then
  echo "Killing nvmf_tgt (PID=$NVMF_PID)..."
  sudo kill -9 $NVMF_PID
else
  echo "nvmf_tgt 未运行"
fi

echo "=== 删除所有 NVMf 子系统 ==="
SUBSYSTEMS=$(sudo /root/spdk/scripts/rpc.py nvmf_get_subsystems | grep -o 'nqn\.[^"]*')
for SUB in $SUBSYSTEMS; do
  echo "删除子系统: $SUB"
  sudo /root/spdk/scripts/rpc.py nvmf_delete_subsystem $SUB || true
done

echo "=== 卸载已挂载的 HugePages ==="
for mountpoint in /mnt/hugepages/data /mnt/hugepages/cache; do
  if mountpoint -q "$mountpoint"; then
    echo "卸载 $mountpoint"
    sudo umount "$mountpoint"
  fi
done

echo "=== 清理 HugePages 目录 ==="
sudo rm -rf /mnt/hugepages/data /mnt/hugepages/cache

echo "=== 杀死 Crail NameNode 和 DataNode ==="
pkill -f "crail namenode" || true
pkill -f "crail datanode" || true

echo "=== 清理完成 ==="
