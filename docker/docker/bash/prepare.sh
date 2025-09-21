#!/bin/bash

set -e

echo "1. 加载内核模块..."
modprobe ib_core ib_cm ib_umad ib_uverbs iw_cm rdma_cm rdma_ucm
modprobe nvme-rdma
modprobe mlx5_core mlx5_ib

echo "2. 配置 HugePages 并挂载目录..."

# 只检测1G HugePages数量
hugepage_path="/sys/kernel/mm/hugepages/hugepages-1048576kB"
if [[ ! -d "$hugepage_path" ]]; then
  echo "错误：系统未启用1G HugePages支持，目录 $hugepage_path 不存在"
  exit 1
fi

available_hugepages=$(cat $hugepage_path/free_hugepages)
total_hugepages=$(cat $hugepage_path/nr_hugepages)

if (( available_hugepages == 0 )); then
  echo "错误：当前系统没有可用的1G HugePages，请先配置"
  echo "当前总量: $total_hugepages, 可用: $available_hugepages"
  exit 1
fi

mountpoint_data=$(mount | grep 'on /mnt/hugepages/data type hugetlbfs' || true)
mountpoint_cache=$(mount | grep 'on /mnt/hugepages/cache type hugetlbfs' || true)

if [[ -z "$mountpoint_data" ]]; then
  mkdir -p /mnt/hugepages/data
  mount -t hugetlbfs -o pagesize=1G nodev /mnt/hugepages/data
  echo "/mnt/hugepages/data 挂载完成"
else
  echo "/mnt/hugepages/data 已挂载"
fi

if [[ -z "$mountpoint_cache" ]]; then
  mkdir -p /mnt/hugepages/cache
  mount -t hugetlbfs -o pagesize=1G nodev /mnt/hugepages/cache
  echo "/mnt/hugepages/cache 挂载完成"
else
  echo "/mnt/hugepages/cache 已挂载"
fi

chmod -R 777 /mnt/hugepages/data /mnt/hugepages/cache

echo "3. 检查 NVMe 设备是否已绑定给 SPDK..."

NVME_BDF="0000:5f:00.0" 
EXPECTED_DRIVER="uio_pci_generic"

if [ -L /sys/bus/pci/devices/$NVME_BDF/driver ]; then
    CURRENT_DRIVER=$(readlink -f /sys/bus/pci/devices/$NVME_BDF/driver | xargs basename)
    if [ "$CURRENT_DRIVER" == "$EXPECTED_DRIVER" ]; then
        echo "设备 $NVME_BDF 已绑定到 $CURRENT_DRIVER，跳过绑定。"
    else
        if [ ! -d "./spdk" ]; then
            echo "未检测到 ./spdk 目录，请将脚本放置在 spdk 根目录或修改路径"
            exit 1
        fi
        cd ./spdk
        ./scripts/setup.sh
        cd ..
    fi
else
    echo "设备 $NVME_BDF 尚未绑定，执行 SPDK 绑定..."
    cd ./spdk && sudo ./scripts/setup.sh
    cd ..
fi


echo "4. 获取 hostnqn..."
nvme gen-hostnqn > ./hostnqn.txt
cat ./hostnqn.txt
echo "hostnqn 已保存至 ./hostnqn.txt"
# 读取新 hostnqn
new_nqn=$(cat ./hostnqn.txt)
# 更新 crail-site.conf 文件
conf_file=./crail_conf/crail-site.conf
key="crail.storage.nvmf.hostnqn"
# 若存在则替换，否则添加（空格分隔格式）
if grep -q "^$key[[:space:]]" "$conf_file"; then
  sed -i "s|^$key[[:space:]].*|$key        $new_nqn|" "$conf_file"
else
  echo "$key        $new_nqn" >> "$conf_file"
fi
echo "已更新 $conf_file 中的 $key 配置项。"


echo "5. 检查 IB 网卡状态..."
IB_IFACE="ibs3f0"
IB_IP="192.168.100.10"

if ip addr show "$IB_IFACE" | grep -q "$IB_IP"; then
    echo "IB 网卡 $IB_IFACE 已配置 IP：$IB_IP"
else
    echo "未检测到 IP，尝试配置 IB 网卡 $IB_IFACE 为 $IB_IP"
    ip addr add "$IB_IP"/24 dev "$IB_IFACE"
    ip link set "$IB_IFACE" up
fi

