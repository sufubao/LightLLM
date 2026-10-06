"""Tune a complete SSM update cycle without touching the serving state pool."""

import torch
import triton

from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, autotune
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_configs():
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def static_key(cache, mode, q, k, v, a, b, cu_seqlens):
    axis = 1 if cu_seqlens is not None else 0
    layers, _, hv, kd, vd = cache.state.shape
    # Keep filenames short, but distinguish packed projections, TP layouts,
    # state precision, recurrent rule and history capacity.
    key = {
        "mode": mode,
        "shape": str((layers, q.shape[-2], hv, cache.num_key_heads, kd, vd, cache.verify_width)),
        "dtype": str((q.dtype, cache.state.dtype)),
        "strides": str((q.stride(axis), k.stride(axis), v.stride(axis), a.stride(0), b.stride(0))),
        "history": cache.capacity if mode == "replay" else 0,
        "gate": cache.lower_bound if mode == "kda" else 0,
        "varlen": cu_seqlens is not None,
    }
    if mode == "replay":
        key["history_dtype"] = str(cache.keys.dtype)
        key["v"] = 4  # Small batches omit the forward fold already handled by prepare.
        key["projection"] = (
            "block_mma_activation_precision"
            if cache.projection_mode == "precompute" and cache.verify_width > 1
            else "inline"
        )
    return key


def run_key(cache, q, cu_seqlens):
    tokens = q.shape[1 if cu_seqlens is not None else 0]
    return triton.cdiv(tokens, cache.verify_width)


def rebuild_inputs(cache, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None):
    batch = run_key(cache, q, cu_seqlens)
    layers, _, hv, kd, vd = cache.state.shape
    state = torch.empty((layers, batch + 1, hv, kd, vd), device=q.device, dtype=cache.state.dtype).normal_(0, 0.02)
    if mode == "replay":
        scratch = type(cache)(
            state,
            cache.capacity,
            cache.verify_width,
            q.dtype,
            num_key_heads=cache.num_key_heads,
            projection_mode=cache.projection_mode,
        )
    else:
        scratch = type(cache)(
            state, cache.verify_width, q.dtype, mode, cache.lower_bound, num_key_heads=cache.num_key_heads
        )
    # Benchmark callbacks must not recursively select a configuration.
    scratch._config_is_fixed = True
    inputs = [
        torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype).normal_(0, 0.2)
        for x in (q, k, v, a, b, a_log, bias)
    ]
    reqs = torch.arange(batch, dtype=torch.int32, device=q.device)
    tokens = q.shape[1 if cu_seqlens is not None else 0]
    cu = (torch.arange(batch + 1, dtype=torch.int32, device=q.device) * cache.verify_width).clamp_max(tokens)
    lengths = cu[1:] - cu[:-1]
    accepted = torch.cat((lengths - 1, lengths.new_zeros(1)))
    partial = accepted.clone()
    partial[:-1] = torch.minimum(reqs % cache.verify_width, lengths - 1)
    snapshot = torch.empty((layers, hv, kd, vd), device=q.device, dtype=state.dtype)
    return (), dict(
        cache=scratch,
        mode=mode,
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        a=inputs[3],
        b=inputs[4],
        a_log=inputs[5],
        bias=inputs[6],
        cu_seqlens=cu if cu_seqlens is not None else None,
        workload=(reqs, accepted, partial, snapshot),
    )


@autotune(
    kernel_name="ssm_update_cycle:v1",
    configs_gen_func=get_configs,
    static_key_func=static_key,
    run_key_func=run_key,
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    rebuild_input_func=rebuild_inputs,
    warmup_all_exist_config=False,
)
def select_config(cache, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None):
    """Return a layout; only rebuilt disposable inputs execute the timed cycle."""
    if workload is None:
        return run_config
    cache.run_config = run_config
    reqs, accepted, partial, snapshot = workload
    if mode == "replay":
        cache.cursors.zero_()
        # Include an entire history period and the subsequent fold. Repeating
        # just the first round would benchmark an empty history forever.
        tokens = q.shape[1 if cu_seqlens is not None else 0]
        rounds = cache.capacity // min(cache.verify_width, tokens) + 1
    else:
        rounds = 2  # Partial acceptance and full acceptance, both across all layers.
    for step in range(rounds):
        positions = cache.prepare_decode(reqs, cu_seqlens)
        for layer in range(cache.state.shape[0]):
            cache.forward(layer, q, k, v, a, b, a_log, bias, reqs, positions, cu_seqlens)
        cache.accept_updates(reqs, partial if mode != "replay" and step == 0 else accepted)
        if mode == "replay" and step == rounds - 2:
            cache._materialize_accepted_state(reqs[:1], snapshot, snapshot=True)
    if mode == "replay":
        cache.merge_accepted_updates(reqs)
    return run_config


def configure_cache(cache, mode, q, k, v, a, b, a_log, bias, cu_seqlens):
    if cache._config_is_fixed or not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        return
    config = select_config(cache, mode, q, k, v, a, b, a_log, bias, cu_seqlens)
    if config is not None:
        cache.run_config = config
        logger.info(f"SSM {mode} update-cycle config: {config}, warmup requests: {run_key(cache, q, cu_seqlens)}")
    # One layout per cache, selected at the largest graph warmup.
    # Per-batch tuning needs explicit layout propagation through verify/accept,
    # including graph padding and overlapping microbatches; never use "last config".
    cache._config_is_fixed = True
