from types import SimpleNamespace
import torch

from lightllm.core.contracts.state import StateView
from lightllm.common.attention_state.bounded_att.base_state import BoundedAttentionState


class CompressedWindowState(BoundedAttentionState):
    def __init__(self, declaration, spec, capacity, device):
        self.declaration = declaration
        self.config = SimpleNamespace(logical_layer_ids=declaration.consumers)
        self.request_capacity = capacity
        self.window = torch.zeros(capacity, spec.window_size, spec.head_dim, dtype=spec.dtype, device=device)
        partial = spec.ratio if spec.layer_id == spec.kv_source and spec.ratio > 1 else 0
        self.partial = torch.zeros(capacity, partial, spec.head_dim, dtype=torch.float32, device=device)
        self.gate = torch.zeros_like(self.partial)

    def get_state_type(self):
        return "shared_compressed_window"

    def contains_logical_layer(self, layer):
        return layer in self.declaration.consumers

    def get_layer_view(self, layer):
        if not self.contains_logical_layer(layer):
            raise KeyError(layer)
        return self.window, self.partial, self.gate

    def reset_bounded_att_state(self, req_indices):
        for tensor in (self.window, self.partial, self.gate):
            tensor[req_indices] = 0

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        if big_page_capacity or small_page_capacity:
            raise ValueError("compressed-window snapshots are not supported")

    def get_cpu_transfer_regions(self):
        return ()

    def get_pd_transfer_regions(self):
        raise ValueError("compressed-window PD transfer is not supported")

    def _unsupported(self, *args, **kwargs):
        raise ValueError("compressed-window cache snapshots, migration and speculative commit are not supported")

    save_bounded_att_runtime_to_pages = _unsupported
    restore_bounded_att_runtime_from_pages = _unsupported
    commit_bounded_att_state = _unsupported
    write_state_pages_to_cpu = _unsupported
    read_state_pages_from_cpu = _unsupported
    copy_big_pages_from_dp = _unsupported
    copy_runtime_state_from_dp = _unsupported
    write_to_pd_page = _unsupported
    read_from_pd_page = _unsupported


class CompressedWindowStore:
    def __init__(self, declarations, fmt, manager, runtime, specs):
        self.format, self.resources = fmt, {}
        for declaration in declarations:
            state = CompressedWindowState(
                declaration, specs[declaration.producer], manager._request_capacity, runtime.device
            )
            manager.register_state(declaration.role, state, resource_id=f"layer_{declaration.producer}")
            self.resources[declaration.key] = declaration, state

    def views(self):
        return {
            key: StateView(decl, self.format, state.get_layer_view(decl.producer))
            for key, (decl, state) in self.resources.items()
        }


class CompressedHistoryStore:
    def __init__(self, declarations, fmt, manager, runtime, specs):
        from lightllm.common.attention_state.full_att.mla import MLAConfig, create_mla_state

        self.format, self.resources = fmt, {}
        for declaration in declarations:
            state = create_mla_state(
                MLAConfig((declaration.consumers,), declaration.shape[0], fmt.dtype, *fmt.shard),
                page_size=fmt.page_size,
                device="meta",
            )
            manager.register_state(declaration.role, state, resource_id=declaration.resource.replace(".", "_"))
            self.resources[declaration.key] = declaration, state

    def views(self):
        return {
            key: StateView(decl, self.format, (state.get_layer_view(decl.producer).squeeze(1),))
            for key, (decl, state) in self.resources.items()
        }
