#!/bin/bash

# ========== User Config (edit here) ==========
# 镜像名与容器名（默认一致，便于统一修改）
IMAGE_NAME="kvcache"
CONTAINER_NAME="$IMAGE_NAME"

# 挂载的模型权重/检查点目录（宿主机 -> 容器）
CKPT_HOST_DIR="/home/workspace/ckpt"
CKPT_MOUNT_TARGET="/tmp/ckpt"

# 其他 docker 参数（GPU/网络/特权等）
DOCKER_EXTRA_ARGS=(
  --gpus all
  --network host
  --privileged
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
  "$IMAGE_NAME" \ 
  /bin/bash
