import collections
import math

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att import ssm_autotune as tuning
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.common import kernel_config as kernel_config_module
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType


def make_inputs(mode, dtype=torch.bfloat16, device="cpu", width=4, projection_mode="inline", kda=False):
    layers, slots, batch, h, hv, kd, vd = 2, 9, 3, 2, 4, 32, 64
    state = torch.randn(layers, slots, hv, kd, vd, dtype=dtype, device=device) * 0.01
    if mode == "replay":
        cache = ReplaySSMCache(state, 8, width, num_key_heads=h, projection_mode=projection_mode, kda=kda)
    else:
        cache = CompactSSMCache(state, width, torch.bfloat16, num_key_heads=h, kda=kda)
    tokens = batch * width
    packed = torch.randn(1, tokens, 2 * h * kd + hv * vd, dtype=torch.bfloat16, device=device)
    q, k, v = [
        x.view(1, tokens, heads, dim)
        for x, heads, dim in zip(packed.split([h * kd, h * kd, hv * vd], -1), [h, h, hv], [kd, kd, vd])
    ]
    gates = torch.randn(tokens, hv * (kd + 1) if kda else hv * 2, device=device, dtype=q.dtype)
    a, b = gates[:, :-hv], gates[:, -hv:]
    return dict(
        cache=cache,
        mode=mode,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=torch.randn(hv, device=device) * 0.1,
        bias=torch.randn(a.shape[-1], device=device) * 0.1,
        cu_seqlens=torch.arange(batch + 1, device=device, dtype=torch.int32) * width,
    )


def tensors(cache):
    return {name: x for name, x in vars(cache).items() if isinstance(x, torch.Tensor)}


def test_replay_projection_has_separate_autotune_cache():
    inline = make_inputs("replay")
    precompute = make_inputs("replay", projection_mode="precompute")
    assert tuning.select_config._static_key(**inline) != tuning.select_config._static_key(**precompute)
    _, rebuilt = tuning.rebuild_inputs(**precompute)
    assert rebuilt["cache"].projection_mode == "precompute"
    assert rebuilt["cache"].keys.dtype == precompute["cache"].keys.dtype == torch.bfloat16
    assert tuning.select_config._static_key(**precompute) == tuning.select_config._static_key(**rebuilt)
    key = tuning.select_config._static_key(**precompute)
    legacy = dict(key, v=2)
    assert KernelConfigs.get_config_file_name(key) != KernelConfigs.get_config_file_name(legacy)
    assert len(KernelConfigs.get_config_file_name(key).encode()) <= 255
    precompute["cache"].keys = precompute["cache"].keys.float()
    assert tuning.select_config._static_key(**precompute) != key


@pytest.fixture(autouse=True)
def isolate_tuner(monkeypatch, tmp_path):
    torch.manual_seed(42)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: 1)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(kernel_config_module, "get_current_device_name", lambda: "NVIDIA H200")
    tuner = tuning.select_config
    monkeypatch.setattr(tuner, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(tuner, "cached_configs", {})
    monkeypatch.setattr(tuner, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(tuner, "warmuped_configs_set", set())


@pytest.mark.parametrize("mode", ["gdn", "replay"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_rebuild_has_real_requests_and_independent_packed_inputs(mode, dtype, kda):
    inputs = make_inputs(mode, dtype, kda=kda)
    cache = inputs["cache"]
    for x in tensors(cache).values():
        x.fill_(1)
    args, rebuilt = tuning.rebuild_inputs(**inputs)
    assert not args
    scratch = rebuilt["cache"]
    assert scratch.state.shape[:2] == (2, 4)  # Active batch + HOLD, not the serving pool.
    assert scratch._config_is_fixed
    assert rebuilt["workload"][0].tolist() == [0, 1, 2]
    assert rebuilt["workload"][1].tolist() == [3, 3, 3, 0]
    assert rebuilt["workload"][2].tolist() == [0, 1, 2, 0]
    for name in ("q", "k", "v", "a", "b", "a_log", "bias"):
        assert rebuilt[name].stride() == inputs[name].stride()
        assert rebuilt[name].data_ptr() != inputs[name].data_ptr()
    assert tuning.select_config._static_key(**inputs) == tuning.select_config._static_key(**rebuilt)
    assert tuning.select_config._run_key(**inputs) == tuning.select_config._run_key(**rebuilt)
    for x in tensors(scratch).values():
        x.zero_()
    for x in tensors(cache).values():
        assert torch.all(x == 1)
    key = tuning.select_config._static_key(**inputs)
    assert len(KernelConfigs.get_config_file_name(key).encode()) <= 255
    other = make_inputs(mode, torch.float32 if dtype == torch.bfloat16 else torch.bfloat16, kda=kda)
    assert tuning.select_config._static_key(**other) != key
    other = make_inputs(mode, dtype, width=3, kda=kda)
    assert tuning.select_config._static_key(**other) != key
    inputs["q"] = inputs["q"].contiguous()
    assert tuning.select_config._static_key(**inputs) != key


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_layout_is_fixed_before_graph_capture_and_cached_configs_do_not_execute(monkeypatch, level):
    inputs = make_inputs("gdn")
    cache = inputs["cache"]
    config = {"BV": 16, "num_warps": 1, "num_stages": 2}
    tuner = tuning.select_config
    key = autotuner_module.frozendict(tuner._static_key(**inputs))
    tuner.cached_configs[key] = {str(tuner._run_key(**inputs)): config}
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: level)
    monkeypatch.setattr(tuner, "_bench", lambda *args, **kwargs: pytest.fail("cached layout must not benchmark"))
    default = cache.run_config.copy()
    with Autotuner.autotune_warmup(AutotuneKernelType.GENERAL):
        tuning.configure_cache(**inputs)
    assert cache.run_config == default and not cache._config_is_fixed
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        tuning.configure_cache(**inputs)
    assert cache.run_config == (default if level == 3 else config)
    assert cache._config_is_fixed
    # Later/smaller graphs, captures, eager calls and acceptance share this layout.
    monkeypatch.setattr(tuning, "select_config", lambda *args, **kwargs: pytest.fail("layout changed after capture"))
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        tuning.configure_cache(**inputs)
    assert not tuner.warmuped_configs_set


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "mode,width,projection_mode",
    [
        ("gdn", 4, "inline"),
        ("replay", 4, "inline"),
        ("replay", 1, "inline"),
        ("replay", 4, "precompute"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kda", [False, True], ids=["gdn", "kda"])
def test_joint_tuning_isolated_state_and_graph_replay(monkeypatch, mode, dtype, width, projection_mode, kda):
    inputs = make_inputs(mode, dtype, "cuda", width=width, projection_mode=projection_mode, kda=kda)
    if width == 1:
        for name in ("q", "k", "v"):
            inputs[name] = inputs[name].transpose(0, 1)
        inputs["cu_seqlens"] = None
    cache = inputs["cache"]
    for name, x in tensors(cache).items():
        if name != "state":
            x.zero_()
    before = {name: x.clone() for name, x in tensors(cache).items()}
    tuner = tuning.select_config
    configs = [{"BV": 16, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 2, "num_stages": 2}]
    monkeypatch.setattr(tuner, "configs_gen_func", lambda: configs)
    bench = tuner._bench
    timings = []

    def checked_bench(*args, **kwargs):
        assert kwargs["cache"] is not cache
        elapsed = bench(*args, **kwargs)
        assert math.isfinite(elapsed), f"failed {mode} {kwargs['run_config']}"
        timings.append(elapsed)
        for name, x in tensors(cache).items():
            torch.testing.assert_close(x, before[name], atol=0, rtol=0)
        return elapsed

    monkeypatch.setattr(tuner, "_bench", checked_bench)
    args = [inputs[name] for name in ("q", "k", "v", "a", "b", "a_log", "bias")]
    reqs = torch.full((3,), cache.hold, device="cuda", dtype=torch.int32)
    cache.prepare_decode(reqs, inputs["cu_seqlens"])
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        # Exercise the real forward hook with the HOLD-only graph warmup input.
        cache.forward(0, *args, reqs, inputs["cu_seqlens"])
    assert len(timings) == len(configs)
    assert cache.run_config in configs
    # An explicit identical layout supplies a reference for capture/replay and
    # alternating real/HOLD rows. It must never trigger its own tuning.
    expected = make_inputs(mode, dtype, "cuda", width=width, projection_mode=projection_mode, kda=kda)["cache"]
    expected.run_config = cache.run_config.copy()
    expected._config_is_fixed = True
    expected.state.copy_(before["state"])
    accepted = torch.zeros(cache.state.shape[1], device="cuda", dtype=torch.int32)

    def step(target):
        target.prepare_decode(reqs, inputs["cu_seqlens"])
        out = [target.forward(layer, *args, reqs, inputs["cu_seqlens"]) for layer in range(target.state.shape[0])]
        target.accept_updates(reqs, accepted)
        return out

    reqs.fill_(cache.hold)
    step(cache)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_out = step(cache)
    reqs.copy_(torch.tensor([2, 0, cache.hold], device="cuda", dtype=torch.int32))
    for iteration in range(12):
        accepted[0] = iteration % width
        accepted[2] = (iteration + 1) % width
        graph.replay()
        expected_out = step(expected)
        for actual, ref in zip(actual_out, expected_out):
            torch.testing.assert_close(actual, ref, atol=0, rtol=0)
        torch.testing.assert_close(cache.state, expected.state, atol=0, rtol=0)
        if mode == "replay":
            cursor = cache.cursors.clone()
            snapshot = cache.snapshot_accepted_state(0)
            torch.testing.assert_close(cache.cursors, cursor, atol=0, rtol=0)
            torch.testing.assert_close(snapshot, expected.snapshot_accepted_state(0), atol=0, rtol=0)
    cache.merge_accepted_updates(reqs[:2])
    expected.merge_accepted_updates(reqs[:2])
    torch.testing.assert_close(cache.state, expected.state, atol=0, rtol=0)
    torch.testing.assert_close(cache.state[:, cache.hold], before["state"][:, cache.hold], atol=0, rtol=0)
    assert len(timings) == len(configs)


@pytest.mark.parametrize("mode", ["gdn", "replay"])
def test_kda_rule_and_gate_bound_separate_tuning_cache(mode):
    inputs = make_inputs(mode, kda=True)
    key = tuning.select_config._static_key(**inputs)
    assert key != tuning.select_config._static_key(**make_inputs(mode))
    _, rebuilt = tuning.rebuild_inputs(**inputs)
    assert rebuilt["cache"].kda
    assert rebuilt["cache"].lower_bound == inputs["cache"].lower_bound
    assert key == tuning.select_config._static_key(**rebuilt)
    inputs["cache"].lower_bound = -2.5
    assert key != tuning.select_config._static_key(**inputs)
