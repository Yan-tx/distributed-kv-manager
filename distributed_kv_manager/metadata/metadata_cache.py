from collections import OrderedDict
import time
import threading
import logging

# 设置日志
logger = logging.getLogger("MetadataCache")
logger.setLevel(logging.DEBUG)

class MetadataCache:
    def __init__(self, meta_manager, pool2_layers=None, pool3_size=1000, pool1_capacity=1000, pool2_capacity=500):
        self.meta_manager = meta_manager  # KVMetadataManager 对象

        # Pool1: session_id -> OrderedDict(key -> KVMetadata)
        self.pool1 = {}                    
        self.pool1_capacity = pool1_capacity

        # Pool2: layer_id -> OrderedDict(key -> KVMetadata)
        self.pool2 = {}                    
        self.pool2_layers = set(pool2_layers or range(30))  
        self.pool2_capacity = pool2_capacity

        # Pool3: LRU for recent accesses
        self.pool3 = OrderedDict()         
        self.pool3_size = pool3_size       

        # 缓存统计信息
        self.stats = {
            'total_queries': 0,
            'pool1_hits': 0,
            'pool2_hits': 0,
            'pool3_hits': 0,
            'etcd_hits': 0,
            'misses': 0,
            'puts': 0,
            'last_reset': time.time()
        }

        # 异步写回队列
        self._write_queue = []
        self._lock = threading.Lock()
        self._stop_flag = False
        self._thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._thread.start()

    def get_metadata(self, key, layer_id=None, session_id=None):
        self.stats['total_queries'] += 1
        key_str = self._key_str(key)
        
        # 记录查询
        logger.debug(f"查询元数据: key={key_str}, layer_id={layer_id}, session_id={session_id}")
        
        # 1. Pool1
        if session_id:
            session_id_str = self._key_str(session_id)
            if session_id_str in self.pool1 and key_str in self.pool1[session_id_str]:
                meta = self.pool1[session_id_str][key_str]
                self._insert_pool3(meta)
                self.stats['pool1_hits'] += 1
                logger.debug(f"Pool1 命中: {key_str}")
                return meta

        # 2. Pool2
        if layer_id is not None and layer_id in self.pool2:
            if key_str in self.pool2[layer_id]:
                meta = self.pool2[layer_id][key_str]
                self._insert_pool3(meta)
                self.stats['pool2_hits'] += 1
                logger.debug(f"Pool2 命中: {key_str}")
                return meta

        # 3. Pool3
        if key_str in self.pool3:
            meta, _ = self.pool3[key_str]
            self.pool3.move_to_end(key_str)
            self.stats['pool3_hits'] += 1
            logger.debug(f"Pool3 命中: {key_str}")
            return meta

        # 4. 回退 etcd
        meta = self.meta_manager.get_metadata(key_str)
        if meta:
            self._insert_cache(meta)
            self.stats['etcd_hits'] += 1
            logger.debug(f"ETCD 命中: {key_str}")
            return meta
        
        # 5. 未命中
        self.stats['misses'] += 1
        logger.debug(f"未命中: {key_str}")
        return None

    def put_metadata(self, meta):
        """写缓存 + 异步写回 etcd"""
        self.stats['puts'] += 1
        logger.debug(f"添加元数据到缓存: {meta.file_path}, 状态: {meta.status}")
        self._insert_cache(meta)
        with self._lock:
            self._write_queue.append(meta)

    def update_access_time(self, key):
        meta = self.get_metadata(key)
        if meta:
            meta.last_access = int(time.time())
            self._insert_pool3(meta)
            with self._lock:
                self._write_queue.append(meta)

    def get_stats(self):
        """获取缓存统计信息"""
        total_hits = self.stats['pool1_hits'] + self.stats['pool2_hits'] + self.stats['pool3_hits'] + self.stats['etcd_hits']
        hit_rate = total_hits / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0
        
        stats = self.stats.copy()
        stats['hit_rate'] = hit_rate
        stats['uptime'] = time.time() - stats['last_reset']
        
        return stats

    def log_stats(self):
        """记录缓存统计信息到日志"""
        stats = self.get_stats()
        logger.info(
            f"缓存统计 - 查询: {stats['total_queries']}, "
            f"命中率: {stats['hit_rate']:.2%}, "
            f"Pool1命中: {stats['pool1_hits']}, "
            f"Pool2命中: {stats['pool2_hits']}, "
            f"Pool3命中: {stats['pool3_hits']}, "
            f"ETCD命中: {stats['etcd_hits']}, "
            f"未命中: {stats['misses']}, "
            f"添加: {stats['puts']}, "
            f"运行时间: {stats['uptime']:.2f}s"
        )

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_queries': 0,
            'pool1_hits': 0,
            'pool2_hits': 0,
            'pool3_hits': 0,
            'etcd_hits': 0,
            'misses': 0,
            'puts': 0,
            'last_reset': time.time()
        }
        logger.info("缓存统计已重置")

    # ------------------ 内部方法 ------------------
    def _insert_cache(self, meta):
        key_str = self._key_str(meta.file_path)
        session_id_str = self._key_str(meta.session_id)

        # Pool1
        if session_id_str not in self.pool1:
            self.pool1[session_id_str] = OrderedDict()
        pool1_dict = self.pool1[session_id_str]
        pool1_dict[key_str] = meta
        pool1_dict.move_to_end(key_str)
        if len(pool1_dict) > self.pool1_capacity:
            removed_key, _ = pool1_dict.popitem(last=False)
            logger.debug(f"Pool1 已满，移除: {removed_key}")

        # Pool2
        if meta.layer_id in self.pool2_layers:
            if meta.layer_id not in self.pool2:
                self.pool2[meta.layer_id] = OrderedDict()
            pool2_dict = self.pool2[meta.layer_id]
            pool2_dict[key_str] = meta
            pool2_dict.move_to_end(key_str)
            if len(pool2_dict) > self.pool2_capacity:
                removed_key, _ = pool2_dict.popitem(last=False)
                logger.debug(f"Pool2 已满，移除: {removed_key}")

        # Pool3
        self._insert_pool3(meta)

    def _insert_pool3(self, meta):
        key_str = self._key_str(meta.file_path)
        self.pool3[key_str] = (meta, time.time())
        self.pool3.move_to_end(key_str)
        if len(self.pool3) > self.pool3_size:
            removed_key, _ = self.pool3.popitem(last=False)
            logger.debug(f"Pool3 已满，移除: {removed_key}")

    def _key_str(self, key):
        if isinstance(key, bytes):
            return key.hex()
        return str(key)

    # ------------------ 异步写回 etcd ------------------
    def _flush_worker(self):
        while not self._stop_flag:
            time.sleep(0.5)
            self._flush_queue()

    def _flush_queue(self):
        with self._lock:
            queue_copy = self._write_queue
            self._write_queue = []

        if queue_copy:
            logger.debug(f"开始异步写回 {len(queue_copy)} 个元数据到ETCD")
            
        success_count = 0
        for meta in queue_copy:
            try:
                self.meta_manager.put_metadata(meta.file_path, meta)
                success_count += 1
            except Exception as e:
                logger.error(f"写回ETCD失败: {e}")

        if queue_copy:
            logger.debug(f"异步写回完成: {success_count}/{len(queue_copy)} 成功")

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        self._flush_queue()  # 停止前清空队列
        self.log_stats()  # 记录最终统计信息