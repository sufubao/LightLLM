import os
from functools import partial

from lightllm.core.device.context import ExecutionContext


def validate_config(config):
    if config is None:
        return
    if config.hardware_platform != "musa":
        raise ValueError("the MUSA execution factory requires --hardware_platform musa")
    if config.run_mode != "normal" or config.mtp_step or config.mtp_mode is not None or config.diverse_mode:
        raise ValueError("MUSA currently supports normal serving without speculative or diverse decoding")
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
            raise ValueError(f"MUSA does not yet support --{name}")
    if not config.disable_symm_mem_allreduce or not config.disable_flashinfer_allreduce:
        raise ValueError("MUSA requires --disable_symm_mem_allreduce and --disable_flashinfer_allreduce; TP uses MCCL")
    if config.llm_kv_type != "None" or config.quant_type not in (None, "none"):
        raise ValueError("MUSA currently supports unquantized weights and KV only")
    if config.sampling_backend != "triton" or config.penalty_counter_mode == "pin_mem_counter":
        raise ValueError("MUSA requires Triton sampling with gpu_counter or cpu_counter penalties")
    if os.environ.get("LIGHTLLM_TRITON_AUTOTUNE_LEVEL", "0") != "0":
        raise ValueError("MUSA kernel autotuning requires LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0")


def create_context(*, index=0, config=None):
    validate_config(config)
    from lightllm.core.device.musa import MusaRuntime
    from lightllm.core.compute.attention.dense.musa import MusaAttentionBackend
    from lightllm.core.compute.attention.dense.backend import DenseStaticParams
    from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.core.compute.linear.backend import LinearBackend
    from lightllm.core.formats.kv.dense import DenseKVWriter
    from lightllm.core.storage.attention import DenseStateStore
    from lightllm.core.registry import ProviderRegistry, AttentionProvider, StateProvider

    providers = ProviderRegistry(
        attention=(AttentionProvider(MusaAttentionBackend, ("prefill", "decode")),),
        storage=(
            StateProvider(
                "token_head_dim",
                "history",
                lambda declarations, format, manager, runtime, specs: DenseStateStore(
                    declarations, format, manager, kv_type="None"
                ),
                DenseKVWriter,
                supports=lambda format: format.encoding == "plain",
            ),
        ),
        parameters=((DenseAttentionSpec, DenseStaticParams.from_weights),),
        linear=(partial(LinearBackend, "torch.linear"),),
    )
    return ExecutionContext(MusaRuntime(index), providers)
