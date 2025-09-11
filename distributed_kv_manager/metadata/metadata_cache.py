from collections import OrderedDict
import time
import threading

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

        # 异步写回队列
        self._write_queue = []
        self._lock = threading.Lock()
        self._stop_flag = False
        self._thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._thread.start()

    def get_metadata(self, key, layer_id=None, session_id=None):
        key_str = self._key_str(key)
        # 1. Pool1
        if session_id:
            session_id_str = self._key_str(session_id)
            if session_id_str in self.pool1 and key_str in self.pool1[session_id_str]:
                meta = self.pool1[session_id_str][key_str]
                self._insert_pool3(meta)
                return meta

        # 2. Pool2
        if layer_id is not None and layer_id in self.pool2:
            if key_str in self.pool2[layer_id]:
                meta = self.pool2[layer_id][key_str]
                self._insert_pool3(meta)
                return meta

        # 3. Pool3
        if key_str in self.pool3:
            meta, _ = self.pool3[key_str]
            self.pool3.move_to_end(key_str)
            return meta

        # 4. 回退 etcd
        meta = self.meta_manager.get_metadata(key_str)
        if meta:
            self._insert_cache(meta)
        return meta

    def put_metadata(self, meta):
        """写缓存 + 异步写回 etcd"""
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
            pool1_dict.popitem(last=False)

        # Pool2
        if meta.layer_id in self.pool2_layers:
            if meta.layer_id not in self.pool2:
                self.pool2[meta.layer_id] = OrderedDict()
            pool2_dict = self.pool2[meta.layer_id]
            pool2_dict[key_str] = meta
            pool2_dict.move_to_end(key_str)
            if len(pool2_dict) > self.pool2_capacity:
                pool2_dict.popitem(last=False)

        # Pool3
        self._insert_pool3(meta)

    def _insert_pool3(self, meta):
        key_str = self._key_str(meta.file_path)
        self.pool3[key_str] = (meta, time.time())
        self.pool3.move_to_end(key_str)
        if len(self.pool3) > self.pool3_size:
            self.pool3.popitem(last=False)

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

        for meta in queue_copy:
            try:
                self.meta_manager.put_metadata(meta.file_path, meta)
            except Exception as e:
                print("Flush error:", e)

    def stop(self):
        self._stop_flag = True
        self._thread.join()
        self._flush_queue()  # 停止前清空队列
