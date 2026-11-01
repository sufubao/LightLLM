import torch
from lightllm.common.attention_state.types import TensorField


class BoundedStorage:
    def __init__(self, spec, request_capacity, mtp_step, device, page_pool_factory):
        self.spec = spec
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.layer_indices = {layer: index for index, layer in enumerate(spec.logical_layers)}
        self.buffers = {
            field.name: torch.zeros(field.shape, dtype=field.dtype, device=device) for field in spec.runtime_buffers
        }
        self.page_pool_factory = page_pool_factory
        self.big_pages_pool = None
        self.small_pages_pool = None

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        assert self.big_pages_pool is None and self.small_pages_pool is None, "page pools already initialized"
        assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
        if self.page_pool_factory is None:
            return
        pools = [
            self.page_pool_factory(size=capacity, device=device, pin_memory=pin_memory) if capacity else None
            for capacity in (big_page_capacity, small_page_capacity)
        ]
        self.big_pages_pool, self.small_pages_pool = pools

    def pool(self, page_kind):
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled"
        return pool

    def page_fields(self, pool, slot):
        return {
            field.name: TensorField(getattr(pool, field.attribute)[slot], field.axes)
            for field in self.spec.snapshot_fields
        }
