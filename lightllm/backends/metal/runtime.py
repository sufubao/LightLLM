"""PyTorch MPS runtime on Apple's single default Metal command queue.

Stream scopes are serialized: events perform a real device synchronization.
CUDA graph capture and IPC are deliberately unavailable on this runtime.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import time
import torch


class MetalEvent:
    def __init__(self, **kwargs):
        self._recorded = False
        self._time = None

    def record(self, stream=None):
        torch.mps.synchronize()
        self._recorded = True
        self._time = time.perf_counter()

    def synchronize(self):
        torch.mps.synchronize()

    def query(self):
        return self._recorded

    def wait(self, stream=None):
        self.synchronize()

    def elapsed_time(self, end_event):
        return (end_event._time - self._time) * 1000


class MetalStream:
    def synchronize(self):
        torch.mps.synchronize()

    def wait_event(self, event):
        event.synchronize()

    def wait_stream(self, stream):
        stream.synchronize()

    def record_event(self, event=None):
        event = event or MetalEvent()
        event.record(self)
        return event


class MetalRuntime:
    name = "metal"
    device_type = "mps"
    device = "mps"
    distributed_backend = "gloo"
    supports_graphs = False
    supports_tensor_cache = False
    supports_pinned_memory = False
    event_type = MetalEvent
    stream_type = MetalStream
    # Used in annotations only. Capture methods fail explicitly.
    graph_type = object
    graph_context_type = object

    def __init__(self):
        self._stream = MetalStream()

    def device_at(self, index):
        if int(index) != 0:
            raise ValueError("Metal supports one device (index 0)")
        return self.device

    def move(self, value, device=None, non_blocking=False, memory_format=torch.preserve_format):
        if device is not None and device not in (0, "mps", "mps:0", torch.device("mps")):
            raise ValueError(f"Unsupported Metal device: {device}")
        return value.to(device=self.device, non_blocking=False, memory_format=memory_format)

    def is_device_tensor(self, value):
        return value.device.type == "mps"

    def generator(self, *args, **kwargs):
        # MPS has no per-device torch.Generator. Sampling uses the default MPS
        # generator seeded by manual_seed, without passing this CPU generator.
        kwargs.pop("device", None)
        return torch.Generator(*args, **kwargs)

    def exponential_(self, tensor, generator=None):
        if generator is None:
            return tensor.exponential_()
        noise = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        noise.exponential_(generator=generator)
        return tensor.copy_(noise)

    def is_available(self):
        return torch.backends.mps.is_available()

    def init(self):
        if not self.is_available():
            raise RuntimeError("Apple Metal requires a macOS arm64 PyTorch build with MPS available")

    lazy_init = init

    def is_initialized(self):
        return self.is_available()

    def current_device(self):
        return 0

    def set_device(self, index):
        self.device_at(index)
        self.init()

    def device_count(self):
        return int(self.is_available())

    def get_device_name(self, device=None):
        return "Apple Metal"

    def get_device_capability(self, device=None):
        # CUDA compute capabilities have no equivalent on Metal.
        return (0, 0)

    def get_device_properties(self, device=None):
        return SimpleNamespace(name=self.get_device_name(), total_memory=torch.mps.recommended_max_memory())

    def synchronize(self, device=None):
        torch.mps.synchronize()

    def current_stream(self, device=None):
        return self._stream

    def stream_scope(self, stream):
        return nullcontext(self._stream)

    def new_stream(self, **kwargs):
        return self._stream

    def new_event(self, **kwargs):
        return MetalEvent(**kwargs)

    def device_scope(self, device):
        self.device_at(device if isinstance(device, int) else 0)
        return nullcontext()

    def new_graph(self, *args, **kwargs):
        raise NotImplementedError("Graph capture is unavailable on Metal")

    capture_graph = new_graph
    graph_pool_handle = new_graph

    def is_current_stream_capturing(self):
        return False

    def empty_cache(self):
        torch.mps.empty_cache()

    def memory_allocated(self, device=None):
        return torch.mps.current_allocated_memory()

    def memory_reserved(self, device=None):
        return torch.mps.driver_allocated_memory()

    def max_memory_reserved(self, device=None):
        raise NotImplementedError("MPS does not expose peak reserved memory statistics")

    def reset_peak_memory_stats(self, device=None):
        raise NotImplementedError("MPS does not expose peak allocator statistics")

    def mem_get_info(self, device=None):
        total = torch.mps.recommended_max_memory()
        return max(0, total - torch.mps.driver_allocated_memory()), total

    def ipc_collect(self):
        raise NotImplementedError("CUDA IPC is unavailable on Metal")

    def manual_seed(self, seed):
        torch.mps.manual_seed(seed)

    manual_seed_all = manual_seed

    def is_bf16_supported(self):
        import platform

        return tuple(map(int, platform.mac_ver()[0].split(".")[:1])) >= (14,)

    def range_start(self, name):
        return None

    def range_end(self, marker):
        return None


def _factory(name):
    def call(self, *args, **kwargs):
        device = kwargs.get("device")
        if isinstance(device, int):
            kwargs["device"] = self.device_at(device)
        # Pinning is a CUDA host transfer optimization; MPS uses unified memory.
        kwargs.pop("pin_memory", None)
        return getattr(torch, name)(*args, **kwargs)

    call.__name__ = name
    return call


for _name in (
    "empty zeros ones full arange tensor randn randint rand empty_like zeros_like "
    "ones_like full_like randn_like rand_like eye linspace logspace as_tensor"
).split():
    setattr(MetalRuntime, _name, _factory(_name))
