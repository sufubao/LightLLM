from abc import ABC, abstractmethod


class KVWriter(ABC):
    def __init__(self, config):
        self.config = config

    def prepare(self, device):
        pass

    @abstractmethod
    def layer_view(self, storage, logical_layer_id):
        ...

    @abstractmethod
    def write(self, storage, logical_layer_id, token_indices, kv):
        ...


class BoundedRuntime:
    def __init__(self, config):
        self.config = config

    def prepare(self, storage):
        pass

    def after_restore(self, storage, req_indices):
        pass

    def check_dp_compatible(self, source, destination):
        assert source.mtp_step == destination.mtp_step, "DP ring copy requires equal MTP capacity"
