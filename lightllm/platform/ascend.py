import os
from functools import partial

from lightllm.core.device.context import ExecutionContext


def validate_config(config):
    if config is None:
        return
    if not config.enable_torch_fallback:
        raise ValueError("Ascend requires --enable_torch_fallback for legacy weight construction")
    if config.page_size % 32 or not 32 <= config.page_size <= 512:
        raise ValueError("Ascend requires --page_size divisible by 32 in [32,512]")
    if config.hardware_platform != "ascend":
        raise ValueError("the Ascend execution factory requires --hardware_platform ascend")
    if config.run_mode != "normal" or config.mtp_step or config.mtp_mode is not None or config.diverse_mode:
        raise ValueError("Ascend currently supports normal serving without speculative or diverse decoding")
    for name in (
        "enable_cpu_cache",
        "enable_dp_prompt_cache_fetch",
        "enable_ep_moe",
        "enable_torch_memory_saver",
        "enable_decode_microbatch_overlap",
        "enable_prefill_microbatch_overlap",
        "enable_prefill_cudagraph",
        "enable_dp_prefill_balance",
        "enable_multimodal",
    ):
        if getattr(config, name, False):
            raise ValueError(f"Ascend does not yet support --{name}")
    if not config.disable_symm_mem_allreduce or not config.disable_flashinfer_allreduce:
        raise ValueError(
            "Ascend requires --disable_symm_mem_allreduce and --disable_flashinfer_allreduce; TP uses HCCL"
        )
    if config.llm_kv_type != "None" or config.quant_type not in (None, "none"):
        raise ValueError("Ascend currently supports unquantized weights and KV only")
    if config.sampling_backend != "triton" or config.penalty_counter_mode == "pin_mem_counter":
        raise ValueError("Ascend requires Triton sampling with gpu_counter or cpu_counter penalties")
    if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", "0") != "0":
        raise ValueError("Ascend kernel autotuning requires LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0")


def create_context(*, index=0, config=None):
    validate_config(config)
    from lightllm.core.device.ascend import AscendRuntime
    from lightllm.core.compute.attention.dense.ascend import AscendAttentionBackend
    from lightllm.core.compute.linear.backend import LinearBackend
    from lightllm.core.formats.kv.ascend import AscendKVWriter
    from lightllm.core.storage.ascend import AscendStateStore
    from lightllm.core.registry import ProviderRegistry, AttentionProvider, StateProvider

    from lightllm.core.compute.pointwise.ascend import bind_rms_norm, bind_rope, bind_swiglu
    from lightllm.common.basemodel.ascend_weights import finalize_weights

    providers = ProviderRegistry(
        attention=(AttentionProvider(AscendAttentionBackend, ("prefill", "decode")),),
        storage=(
            StateProvider(
                "token_head_dim",
                "history",
                lambda declarations, format, manager, runtime, specs: AscendStateStore(declarations, format, manager),
                AscendKVWriter,
                supports=lambda format: format.encoding == "plain",
            ),
        ),
        linear=(partial(LinearBackend, "torch.linear"),),
        weight_finalizers=(finalize_weights,),
        operations=(("rms_norm", bind_rms_norm), ("rope", bind_rope), ("swiglu", bind_swiglu)),
    )
    return ExecutionContext(AscendRuntime(index), providers)
