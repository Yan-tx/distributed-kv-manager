from typing import TYPE_CHECKING, Union
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

class DistributedKVConnector(KVConnectorBase):
    """
    DistributedKVConnector
    """
    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        """
        engine: DistributedKVEngineBase 子类实例
        """
        from distributed_kv_manager.engine import(
            StoreStatus,RetrieveStatus,init_engine,
            retrieve_kv,should_retrieve,store_kv,should_store)
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self.engine = init_engine(config)
        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config
        self.retrieve_kv = retrieve_kv
        self.should_retrieve = should_retrieve
        self.store_kv = store_kv
        self.should_store = should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus
        self.engine_name = getattr(config, "engine_id", "unknown_engine")

        logger.info(f"DistributedKVConnector initialized with engine {self.engine_name}")

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool, "ModelInputForGPUWithSamplingMetadata"]:

        retrieve_status = self.engine.should_retrieve(model_input)
        hidden_or_intermediate_states, bypass_model_exec, model_input  = self.engine.retrieve_kv(
            model_executable, model_input, kv_caches, retrieve_status
        )
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:

        store_status = self.engine.should_store(model_input)
        self.engine.store_kv(
            self.vllm_config.model_config,
            self.vllm_config.parallel_config,
            # None,None,
            self.transfer_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
            hidden_or_intermediate_states
        )

    def close(self):
        self.engine.destroy_engine(self.engine_name)
        logger.info(f"DistributedKVConnector engine {self.engine_name} destroyed")
