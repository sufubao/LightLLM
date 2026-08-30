import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    Qwen3NextMemManager,
)


class Qwen4ExpMemManager(Qwen3NextMemManager):
    """Qwen3Next cache plus QSA raw/compressed key side state."""

    def __init__(
        self,
        *args,
        qsa_enabled: bool,
        qsa_head_dim: int,
        qsa_rotary_half_dim: int,
        **kwargs,
    ):
        self.qsa_enabled = qsa_enabled
        self.qsa_head_dim = qsa_head_dim
        self.qsa_rotary_half_dim = qsa_rotary_half_dim
        super().__init__(*args, **kwargs)

    def get_cell_size(self):
        cell_size = super().get_cell_size()
        if self.qsa_enabled:
            # One raw and one compressed BF16 index key per full-attention
            # layer.  RoPE rows are shared across layers and accounted below.
            cell_size += (
                2
                * self.layer_num
                * self.qsa_head_dim
                * torch._utils._element_size(self.dtype)
            )
            cell_size += (
                2
                * 3
                * self.qsa_rotary_half_dim
                * torch._utils._element_size(self.dtype)
            )
        return cell_size

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        super()._init_buffers(size, dtype, head_num, head_dim, layer_num)
        if not self.qsa_enabled:
            self.qsa_raw_key_buffer = None
            self.qsa_compressed_key_buffer = None
            self.qsa_position_cos_buffer = None
            self.qsa_position_sin_buffer = None
            return

        self.qsa_raw_key_buffer = torch.empty(
            (layer_num, size + 1, self.qsa_head_dim),
            dtype=dtype,
            device="cuda",
        )
        self.qsa_compressed_key_buffer = torch.empty_like(
            self.qsa_raw_key_buffer
        )
        position_shape = (size + 1, 3, self.qsa_rotary_half_dim)
        self.qsa_position_cos_buffer = torch.empty(
            position_shape, dtype=dtype, device="cuda"
        )
        self.qsa_position_sin_buffer = torch.empty_like(
            self.qsa_position_cos_buffer
        )

    def _qsa_layer_index(self, layer_index: int) -> int:
        if not self.qsa_enabled:
            raise RuntimeError("QSA side cache is disabled for this server")
        return self.linear_config.get_full_att_kv_layer_index(layer_index)

    def get_qsa_raw_key_buffer(self, layer_index: int) -> torch.Tensor:
        return self.qsa_raw_key_buffer[self._qsa_layer_index(layer_index)]

    def get_qsa_compressed_key_buffer(self, layer_index: int) -> torch.Tensor:
        return self.qsa_compressed_key_buffer[self._qsa_layer_index(layer_index)]

    def _free_buffers(self):
        super()._free_buffers()
        self.qsa_raw_key_buffer = None
        self.qsa_compressed_key_buffer = None
        self.qsa_position_cos_buffer = None
        self.qsa_position_sin_buffer = None


__all__ = ["Qwen4ExpMemManager"]
