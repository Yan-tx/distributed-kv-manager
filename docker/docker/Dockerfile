# 基础镜像：使用包含 CUDA 的 PyTorch 镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# 设置清华源并安装依赖
RUN sed -i 's@http://.*archive.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@http://.*security.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    apt update && \
    apt install -y \
        build-essential git wget libtool autoconf apt-utils curl \
        openjdk-11-jdk maven \
        python3 python3-pip libibverbs-dev librdmacm-dev \
        nvme-cli sudo iproute2 pciutils libnuma-dev \
        libaio-dev libssl-dev uuid-dev kmod ninja-build cmake python3-dev \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y cmake && \
    apt clean

# 设置 JAVA 环境
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 安装 Disni
RUN git clone https://github.com/zrlio/disni.git /root/disni && \
    cd /root/disni && \
    mvn -DskipTests install && \
    cd libdisni && \
    ./autoprepare.sh && \
    ./configure --with-jdk=$JAVA_HOME && \
    make && make install

# 安装 SPDK
RUN git clone https://github.com/spdk/spdk.git /root/spdk && \
    cd /root/spdk && \
    git submodule update --init && \
    ./scripts/pkgdep.sh && \
    ./configure --with-rdma && \
    make -j$(nproc)

# 安装 Crail
RUN git clone https://github.com/apache/incubator-crail.git /root/incubator-crail && \
    cd /root/incubator-crail && \
    mvn -DskipTests install && \
    cd assembly/target && \
    tar -zxvf apache-crail-1.3-incubating-SNAPSHOT-bin.tar.gz && \
    mv apache-crail-1.3-incubating-SNAPSHOT /root/crail

# 设置 CRAIL 环境变量
ENV CRAIL_HOME=/root/crail
ENV PATH=$CRAIL_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CRAIL_HOME/lib:/usr/local/lib:$LD_LIBRARY_PATH

# 拷贝 Disni 动态库
RUN mkdir -p $CRAIL_HOME/lib && cp /usr/local/lib/libdisni.so $CRAIL_HOME/lib

# 拷贝脚本和配置文件
COPY run_node.sh /root/
COPY test.sh /root/
COPY clean.sh /root/
RUN chmod +x /root/run_node.sh /root/test.sh /root/clean.sh
COPY crail_conf /root/crail/conf

# 构建 python_crail
COPY crail-example /root/crail-example
RUN cd /root/crail-example && mvn clean package
COPY test_jni_call.py /root

# 安装 Python 依赖（使用国内镜像源）
RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy==1.24.4 sentencepiece==0.1.99 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝 vLLM 源码和 cutlass 子模块（防止编译阶段失败）
RUN git clone -b v0,9,2 /root/vllm
# COPY cutlass /root/vllm/.deps/cutlass-src

# 设置编译环境变量
ENV MAX_JOBS=20
ENV MAKEFLAGS="-j${MAX_JOBS}"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# 安装 vLLM 所需依赖，启用 editable 模式
# COPY wheels /tmp/wheels
RUN pip install \
        setuptools==77.0.1 \
        packaging>=24.2 \
        cmake \
        ninja \
        regex \
        setuptools_scm \
        wheel \
        pytest \
        -i https://pypi.tuna.tsinghua.edu.cn/simple 

#测试元数据时若若不用构建vLLM 可以先注释掉 
RUN pip install -e /root/vllm
RUN pip install transformers==4.53.2

RUN pip install --upgrade pip setuptools wheel packaging
RUN pip install "protobuf>=3.19.0,<3.21" \
                etcd3

# 安装 etcd 如果在线安装失败可以自行下载解压后直接复制
# COPY etcd /root/etcd
# 下载并解压 etcd
RUN mkdir -p /root/etcd && \
    cd /root/etcd && \
    wget -q https://github.com/etcd-io/etcd/releases/download/v3.6.4/etcd-v3.6.4-linux-amd64.tar.gz   && \
    tar -xzf etcd-v3.6.4-linux-amd64.tar.gz --strip-components=1 && \
    rm etcd-v3.6.4-linux-amd64.tar.gz
RUN mkdir -p /tmp/etcd-data

# 启动 etcd 
# RUN nohup /opt/etcd/etcd \
#     --data-dir /tmp/etcd-data \
#     --listen-client-urls http://0.0.0.0:2379 \
#     --advertise-client-urls http://0.0.0.0:2379 \
#     > /tmp/etcd.log 2>&1 &


RUN git cole https://github.com/Yan-tx/distributed-kv-manager.git /root/distributed_kv_manager
RUN pip install -e /root/distributed_kv_manager

# 设置 CRAIL 环境变量
ENV CRAIL_KVCACHE_JAR=/root/crail-example/target/crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar
ENV CRAIL_CONF_DIR=/root/crail/conf

WORKDIR /root
