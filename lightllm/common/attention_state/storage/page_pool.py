from __future__ import annotations
import torch
from lightllm.common.attention_state.bounded_att.conv_ssm_types import ConvSSMStateConfig
from lightllm.common.attention_state.storage.host_memory import create_page_pool_tensor, register_page_pool_tensor


class GDNPagesPool:
    def __init__(
        self,
        size: int,
        config: ConvSSMStateConfig,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        assert size >= 0, "pool size must be nonnegative"
        self.size = size
        device = torch.device(device)
        pin_memory = pin_memory and device.type == "cpu"
        self.conv_state = create_page_pool_tensor(
            size=(size, len(config.logical_layer_ids), config.conv_channels, config.conv_width),
            dtype=config.conv_dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.ssm_state = create_page_pool_tensor(
            size=(size, len(config.logical_layer_ids), config.num_v_heads, config.key_dim, config.value_dim),
            dtype=config.ssm_dtype,
            device=device,
            pin_memory=pin_memory,
        )

    # ========================================================================
    # 跨进程映射：在接收进程的当前 GPU 上注册共享页池
    # ========================================================================

    def register_pin_memory(self) -> None:
        """注册 Conv/SSM 共享存储，供本进程 GPU 使用；重复调用不会重复注册。"""
        register_page_pool_tensor(tensor=self.conv_state)
        register_page_pool_tensor(tensor=self.ssm_state)


class KDAPagesPool:
    def __init__(
        self,
        size: int,
        config: ConvSSMStateConfig,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        assert size >= 0, "pool size must be nonnegative"
        assert (
            config.num_k_heads == config.num_v_heads and config.key_dim == config.value_dim
        ), "KDA pages require equal Q/K/V heads and dimensions"
        self.size = size
        device = torch.device(device)
        pin_memory = pin_memory and device.type == "cpu"
        storage_layer_count, heads, dim = len(config.logical_layer_ids), config.num_k_heads, config.key_dim
        self.conv_state = create_page_pool_tensor(
            size=(size, storage_layer_count, 3 * heads * dim, config.conv_width),
            dtype=config.conv_dtype,
            device=device,
            pin_memory=pin_memory,
        )
        self.ssm_state = create_page_pool_tensor(
            size=(size, storage_layer_count, heads, dim, dim),
            dtype=config.ssm_dtype,
            device=device,
            pin_memory=pin_memory,
        )

    # ========================================================================
    # 跨进程映射：在接收进程的当前 GPU 上注册共享页池
    # ========================================================================

    def register_pin_memory(self) -> None:
        """注册 Conv/SSM 共享存储，供本进程 GPU 使用；重复调用不会重复注册。"""
        register_page_pool_tensor(tensor=self.conv_state)
        register_page_pool_tensor(tensor=self.ssm_state)


class WindowPagesPool:
    def __init__(
        self,
        size: int,
        snapshot_shape: tuple[int, int, int],
        dtype: torch.dtype,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        # [页槽位, 物理层, 时间窗口, 字段宽度]；只保存快照，不保存请求或长度元数据。
        self.buffer = create_page_pool_tensor(
            size=(size, *snapshot_shape),
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )

    def register_pin_memory(self) -> None:
        """跨 DP 导入后，在接收进程当前设备上注册共享页池。"""
        register_page_pool_tensor(tensor=self.buffer)
