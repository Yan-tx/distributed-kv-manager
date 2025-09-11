from enum import Enum

class StoreStatus(Enum):
    STORED = 1
    SKIPPED = 2

class RetrieveStatus(Enum):
    HIT = 1
    MISS = 2

class DistributedKVEngineBase:
    """KV缓存存储后端抽象"""
    def should_store(self, model_input) -> StoreStatus:
        raise NotImplementedError

    def store_kv(self, model_config, parallel_config, transfer_config,
                 model_executable, model_input, kv_caches, store_status):
        raise NotImplementedError

    def should_retrieve(self, model_input) -> RetrieveStatus:
        raise NotImplementedError

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

_engine_instance: DistributedKVEngineBase = None

def init_engine(config):
    global _engine_instance
    if _engine_instance is None:
        backend_type = getattr(config.kv_transfer_config, "backend_type", "crail")
        if backend_type == "crail":
            from .backends import CrailKVEngine
            _engine_instance = CrailKVEngine(config)
        else:
            from .backends import LMCacheEngine
            _engine_instance = LMCacheEngine(config)
    return _engine_instance

def destroy_engine(engine_name=None):
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.close()
        _engine_instance = None

def should_store(model_input) -> StoreStatus:
    return _engine_instance.should_store(model_input)

def store_kv(model_config, parallel_config, transfer_config,
             model_executable, model_input, kv_caches, store_status):
    return _engine_instance.store_kv(model_config, parallel_config,
                                     transfer_config, model_executable,
                                     model_input, kv_caches, store_status)

def should_retrieve(model_input) -> RetrieveStatus:
    return _engine_instance.should_retrieve(model_input)

def retrieve_kv(model_executable, model_input, kv_caches, retrieve_status):
    return _engine_instance.retrieve_kv(model_executable, model_input,
                                        kv_caches, retrieve_status)
