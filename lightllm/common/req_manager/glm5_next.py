from lightllm.common.indexer_state import IndexerStateManager
from lightllm.utils.envs_utils import get_env_start_args

from .linear_att import ReqManagerForMamba


class Glm5NextReqManager(ReqManagerForMamba):
    """KDA runtime state and the NSA indexer's incomplete four-token pool."""

    def __init__(self, max_request_num, max_sequence_length, mem_manager, linear_config):
        # Both checkpoint sizes are multiples of this hash page. A restored
        # prefix therefore has no incomplete K-pool to serialize or replay.
        assert (
            get_env_start_args().linear_att_hash_page_size % linear_config.index_kpool == 0
        ), "GLM K-pool requires cache pages aligned to index_kpool"
        super().__init__(max_request_num, max_sequence_length, mem_manager, linear_config)
        self.indexer_state_manager = IndexerStateManager(max_request_num, linear_config, self.mtp_step)
        self.req_to_indexer_tail = self.indexer_state_manager.state

    def get_indexer_tail_buffer(self, layer_index):
        return self.req_to_indexer_tail.buffer[self.linear_config.get_full_att_kv_layer_index(layer_index)]

    def init_hybrid_attention_state(self, req):
        super().init_hybrid_attention_state(req)
        self.req_to_indexer_tail.buffer[:, req.req_idx].zero_()

    def restore_state(self, req, state_cache_manager, buffer_idx):
        super().restore_state(req, state_cache_manager, buffer_idx)
        self.req_to_indexer_tail.buffer[:, req.req_idx].zero_()
