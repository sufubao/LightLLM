import torch

from lightllm.core.contracts.device import DeviceCaps


class AscendRuntime:
    distributed_backend = "hccl"

    def __init__(self, index=0):
        import torch_npu  # noqa: F401

        self.api = torch.npu
        self.set_device(index)
        self.device = torch.device("npu", index)
        self.caps = DeviceCaps("npu", index, platform="ascend", architecture=self.api.get_device_name(index))

    def set_device(self, index):
        self.api.set_device(index)

    @staticmethod
    def process_group_options():
        return {}

    def memory_info(self):
        return self.api.mem_get_info(self.device)

    def empty_cache(self):
        self.api.empty_cache()

    def workspace_budget(self, mem_fraction):
        self.empty_cache()
        free, total = self.memory_info()
        return max(0, int(free - total * (1 - mem_fraction)))

    def allocate_workspace(self, size):
        return torch.empty(size, dtype=torch.uint8, device=self.device)

    def allocate_tensor(self, shape, dtype):
        from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager

        return g_cache_manager.alloc_tensor(shape, dtype, device=self.device)

    def move_batch(self, batch):
        return batch.to_device(self.device)

    def create_stream(self):
        return self.api.Stream(device=self.device)

    def current_stream(self):
        return self.api.current_stream(self.device)

    def use_stream(self, stream=None):
        return self.api.stream(stream or self.current_stream())

    def create_event(self, *, enable_timing=False):
        return self.api.Event(enable_timing=enable_timing)

    def record_event(self):
        event = self.create_event()
        event.record(self.current_stream())
        return event

    def synchronize(self):
        self.api.synchronize(self.device)

    @staticmethod
    def barrier():
        torch.distributed.barrier()

    def graph_pool(self):
        return self.api.graph_pool_handle()

    def create_graph(self):
        return self.api.NPUGraph()

    def capture_graph(self, graph, *, pool=None, memory_saver=None):
        return self.api.graph(graph, pool=pool)

    @staticmethod
    def decode_warmup_length(max_length):
        return max_length

    @staticmethod
    def borrow_tensor(tensor):
        import torch_npu

        return torch_npu._C._weak_ref_tensor(tensor)
