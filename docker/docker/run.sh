#!/bin/bash

# ========== User Config (edit here) ==========
# 镜像名与容器名（默认一致，便于统一修改）
IMAGE_NAME="crail"
CONTAINER_NAME="$IMAGE_NAME"

# 设备与资源（按需调整）
NVME_DEV="nvme0"                      # NVMe 设备名（/dev/$NVME_DEV）

# 权重/检查点目录挂载（宿主机 -> 容器）
CKPT_HOST_DIR="/home/ms-admin/workspace3/tiexin/ckpt"
CKPT_MOUNT_TARGET="/tmp/ckpt"

# HugePages/内存映射目录（宿主机 -> 容器）
HUGEPAGES_DATA="/mnt/hugepages/data"
HUGEPAGES_CACHE="/mnt/hugepages/cache"
HUGEPAGES_SYS="/sys/kernel/mm/hugepages"
HUGEPAGES_DEV="/dev/hugepages"

# 其他 docker 参数（GPU/网络/特权等）
DOCKER_EXTRA_ARGS=(
  --gpus all
  --network host
  --privileged
  --device=/dev/infiniband
  --device=/dev/$NVME_DEV
)
# =============================================

# 可选：检查是否已构建该镜像（可省略）
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
  echo "Error: Docker image '$IMAGE_NAME' not found. Please build it first."
  exit 1
fi

# 启动容器
docker run -it --rm \
  "${DOCKER_EXTRA_ARGS[@]}" \
  --name "$CONTAINER_NAME" \
  -v "$CKPT_HOST_DIR":"$CKPT_MOUNT_TARGET" \
  -v "$HUGEPAGES_DATA":"$HUGEPAGES_DATA" \
  -v "$HUGEPAGES_CACHE":"$HUGEPAGES_CACHE" \
  -v "$HUGEPAGES_SYS":"$HUGEPAGES_SYS" \
  -v "$HUGEPAGES_DEV":"$HUGEPAGES_DEV" \
  "$IMAGE_NAME"
