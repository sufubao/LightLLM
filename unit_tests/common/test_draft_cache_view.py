from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import mem_manager as mem_module
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import (
    FP8StaticPerHeadQuantQwen3NextMemManager,
    Qwen3NextMemManager,
)
from lightllm.common.state_cache_manager import LinearAttCacheConfig


def make_manager(tp_size=1, dtype=torch.bfloat16, heads=4, head_dim=256, quant=False, device="cpu"):
    cls = FP8StaticPerHeadQuantQwen3NextMemManager if quant else Qwen3NextMemManager
    manager = cls.__new__(cls)
    manager.head_num = max(heads // tp_size, 1)
    manager.head_dim = head_dim
    manager.layer_num = 3
    manager.page_size = 4
    manager.size = 8
    manager.dtype = dtype
    manager.allocator = object()
    manager.HOLD_TOKEN_MEMINDEXES = (8, 9, 10, 11)
    manager.linear_config = LinearAttCacheConfig(
        tp_world_size=tp_size,
        full_att_all_num_kv_heads=heads,
        full_att_dtype=dtype,
        full_att_num_kv_heads=manager.head_num,
        full_att_head_dim=head_dim,
        global_linear_k_heads=8,
        global_linear_v_heads=8,
        num_linear_k_heads=8 // tp_size,
        num_linear_v_heads=8 // tp_size,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=4,
        linear_layer_num=6,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=4,
        all_layer_num=8,
        draft_full_att_kv_layer_num=1,
    )
    manager.kv_buffer = torch.zeros((3, 12, 2 * manager.head_num, head_dim), dtype=dtype, device=device)
    manager.operator = manager.operator_class(manager)
    if quant:
        manager.scales = torch.arange(1, 1 + 3 * 2 * manager.head_num, device=device).float().view(3, -1)
        manager.q_scales = torch.arange(1, 1 + 3 * manager.head_num, device=device).float().view(3, -1)
    return manager


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8])
def test_view_shares_slots_but_preserves_target_layout(tp_size, dtype):
    target = make_manager(tp_size, dtype, quant=dtype == torch.uint8)
    draft = target.get_draft_mem_manager(8, 128)
    assert draft is not target
    assert draft.allocator is target.allocator
    assert draft.HOLD_TOKEN_MEMINDEXES is target.HOLD_TOKEN_MEMINDEXES
    assert draft.operator is not target.operator
    assert draft.operator.mem_manager is draft
    assert target.operator.mem_manager is target
    assert (target.head_num, target.head_dim) == (4 // tp_size, 256)
    assert (draft.head_num, draft.head_dim) == (8 // tp_size, 128)
    assert draft.kv_buffer.data_ptr() == target.kv_buffer.data_ptr()
    assert draft.kv_buffer.is_contiguous()
    # The last physical page is reserved for padding; its slots must also alias.
    k, v = draft.get_att_input_params(8)
    k[11].fill_(3)
    v[11].fill_(7)
    target_k, target_v = target.get_att_input_params(8)
    assert torch.all(target_k[11] == 3)
    assert torch.all(target_v[11] == 7)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_identical_layout_preserves_manager(tp_size):
    target = make_manager(tp_size)
    assert target.get_draft_mem_manager(4, 256) is target


@pytest.mark.parametrize(
    "tp_size,heads,dim,error",
    [
        (8, 8, 128, "different per-rank KV widths"),
        (1, 8, 256, "equal global KV widths"),
        (3, 8, 128, "cannot be sharded or replicated"),
        (1, 0, 128, "cannot be sharded or replicated"),
        (1, 8, 0, "equal global KV widths"),
    ],
)
def test_unsupported_layout_fails_before_rebinding(tp_size, heads, dim, error):
    target = make_manager(tp_size)
    operator, buffer = target.operator, target.kv_buffer
    with pytest.raises(ValueError, match=error):
        target.get_draft_mem_manager(heads, dim)
    assert target.kv_buffer is buffer
    assert target.operator is operator
    assert operator.mem_manager is target


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_fp8_regroups_k_v_and_q_without_mutating_target(tp_size):
    target = make_manager(tp_size, torch.uint8, quant=True)
    old_scales, old_q = target.scales.clone(), target.q_scales.clone()
    draft = target.get_draft_mem_manager(8, 128)
    torch.testing.assert_close(draft.scales, old_scales.repeat_interleave(2, dim=-1))
    torch.testing.assert_close(draft.q_scales, old_q.repeat_interleave(2, dim=-1))
    draft.scales.fill_(99)
    draft.q_scales.fill_(99)
    torch.testing.assert_close(target.scales, old_scales)
    torch.testing.assert_close(target.q_scales, old_q)


def test_fp8_merge_uses_largest_calibrated_range_separately_for_k_and_v():
    target = make_manager(dtype=torch.uint8, heads=8, head_dim=128, quant=True)
    draft = target.get_draft_mem_manager(4, 256)
    torch.testing.assert_close(draft.scales, target.scales.view(3, 2, 4, 2).amax(-1).reshape(3, 8))
    torch.testing.assert_close(draft.q_scales, target.q_scales.view(3, 4, 2).amax(-1))


def test_fp8_requires_complete_calibration_rows():
    target = make_manager(dtype=torch.uint8, quant=True)
    target.scales = target.scales[:2]
    with pytest.raises(ValueError, match="Invalid FP8 calibration scale shape"):
        target.get_draft_mem_manager(8, 128)


def page_io_reference(mem_indexes, page_tensor, kv_buffer, tp_index, tp_world_size, mode):
    local_heads = kv_buffer.shape[2] // 2
    global_heads = page_tensor.shape[2] // 2
    repeat = local_heads * tp_world_size // global_heads
    start = tp_index // repeat * local_heads
    for local_base, global_base in [(0, start), (local_heads, global_heads + start)]:
        for token_idx, mem_idx in enumerate(mem_indexes.tolist()):
            cache = kv_buffer[:, mem_idx, local_base : local_base + local_heads]
            page = page_tensor[token_idx, :, global_base : global_base + local_heads]
            if mode == "write":
                page.copy_(cache)
            else:
                cache.copy_(page)


@pytest.mark.parametrize("source_tp,dest_tp", [(s, d) for s in [1, 2, 4] for d in [1, 2, 4]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")),
    ],
)
def test_target_owned_pd_transport_roundtrips_draft_across_tp(monkeypatch, source_tp, dest_tp, dtype, device):
    if device == "cpu":
        monkeypatch.setattr(mem_module, "page_io", page_io_reference)
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
        tensor = torch.tensor
        monkeypatch.setattr(
            torch,
            "tensor",
            lambda *args, **kwargs: tensor(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}),
        )
    source = [make_manager(source_tp, dtype, device=device) for _ in range(source_tp)]
    dest = [make_manager(dest_tp, dtype, device=device) for _ in range(dest_tp)]
    page = torch.zeros((1, 2, 3, 8, 256), dtype=dtype, device=device)
    for manager in [source[0], dest[0]]:
        manager.kv_move_buffer = page
        manager._buffer_mem_indexes_tensors = [torch.empty(2, dtype=torch.int64, pin_memory=device == "cuda")]
    expected = torch.arange(2 * 16 * 128, device=device).remainder(127).reshape(2, 16, 128).to(dtype)
    for rank, manager in enumerate(source):
        draft = manager.get_draft_mem_manager(8, 128)
        k, v = draft.get_att_input_params(8)
        n = draft.head_num
        k[1:3] = expected[:, rank * n : (rank + 1) * n]
        v[1:3] = expected[:, 8 + rank * n : 8 + (rank + 1) * n]
    source[0].write_mem_to_page_kv_move_buffer([1, 2], 0, 0, source, source_tp)
    dest[0].read_page_kv_move_buffer_to_mem([3, 4], 0, 0, dest, dest_tp)
    for rank, manager in enumerate(dest):
        draft = manager.get_draft_mem_manager(8, 128)
        k, v = draft.get_att_input_params(8)
        n = draft.head_num
        torch.testing.assert_close(k[3:5], expected[:, rank * n : (rank + 1) * n])
        torch.testing.assert_close(v[3:5], expected[:, 8 + rank * n : 8 + (rank + 1) * n])
        assert torch.count_nonzero(manager.kv_buffer[:2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant", [False, True])
def test_draft_operator_writes_and_reads_its_own_layout_on_cuda(quant):
    target = make_manager(dtype=torch.uint8 if quant else torch.bfloat16, quant=quant, device="cuda")
    draft = target.get_draft_mem_manager(8, 128)
    kv = torch.randn((2, 16, 128), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([1, 9], dtype=torch.int32, device="cuda")
    draft.operator.copy_kv_to_mem_manager(8, slots, kv)
    k, v = draft.get_att_input_params(8)
    actual = torch.cat((k[slots.long()], v[slots.long()]), dim=1)
    if quant:
        scales = draft.scales[2].view(1, 16, 1)
        expected = (kv.float() / scales).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        expected = kv
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_target_owned_cpu_cache_restores_draft_bytes(monkeypatch, tp_size, dtype):
    from lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy import (
        copy_kv_buffer_to_cpu_cache,
        copy_cpu_cache_to_kv_buffer,
    )
    from lightllm.common.state_cache_manager import linear_att

    args = SimpleNamespace(linear_att_page_block_num=1, linear_att_hash_page_size=8, cpu_cache_token_page_size=8)
    monkeypatch.setattr(linear_att, "get_env_start_args", lambda: args)
    manager = make_manager(tp_size, dtype, device="cuda")
    config = manager.linear_config
    cpu_pages = torch.zeros((1, config.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)
    conv = torch.zeros((1, 6, *config.get_conv_state_shape()), dtype=torch.bfloat16, pin_memory=True)
    ssm = torch.zeros((1, 6, *config.get_ssm_state_shape()), dtype=torch.float32, pin_memory=True)
    indexes = torch.arange(8, dtype=torch.int32, device="cuda")
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    ready = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    common = dict(
        mem_indexes=indexes,
        page_indexes=zero,
        big_page_buffer_ids=zero,
        cpu_kv_conv_state=conv,
        cpu_kv_ssm_state=ssm,
        cpu_cache_tensor=cpu_pages,
        tp_world_size=tp_size,
        big_page_token_num=8,
        linear_config=config,
    )
    expected = []
    for rank in range(tp_size):
        draft = manager.get_draft_mem_manager(8, 128)
        draft.kv_buffer[:, :8].random_(0, 127)
        expected.append(draft.kv_buffer[:, :8].clone())
        copy_kv_buffer_to_cpu_cache(page_readies=ready, gpu_kv_full_att_state=manager.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
    for rank in range(tp_size):
        manager.kv_buffer.zero_()
        copy_cpu_cache_to_kv_buffer(gpu_full_att_kv_state=manager.kv_buffer, tp_rank=rank, **common)
        torch.cuda.synchronize()
        torch.testing.assert_close(manager.get_draft_mem_manager(8, 128).kv_buffer[:, :8], expected[rank])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant", [False, True])
def test_noncausal_fa3_draft_attention_and_cuda_graph(quant):
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache
    from lightllm.common.basemodel.triton_kernel.quantization.q_per_head_fp8_quant import (
        q_per_head_static_fp8_quant,
    )

    if flash_attn_with_kvcache is None:
        pytest.skip("FA3 is unavailable")
    target = make_manager(dtype=torch.uint8 if quant else torch.bfloat16, quant=quant, device="cuda")
    if quant:
        target.scales.mul_(0.001)
        target.q_scales.mul_(0.001)
    draft = target.get_draft_mem_manager(8, 128)
    generator = torch.Generator(device="cuda").manual_seed(0)
    kv = torch.randn((4, 16, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    q = torch.randn((2, 32, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    slots = torch.arange(4, dtype=torch.int32, device="cuda")
    page_table = slots.view(1, 4)
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    k, v = draft.get_att_input_params(8)
    kwargs = dict(
        page_table=page_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=cu_q,
        max_seqlen_q=2,
        causal=False,
    )
    if quant:
        k = k.view(torch.float8_e4m3fn)
        v = v.view(torch.float8_e4m3fn)
        kwargs.update(
            q_descale=draft.q_scales[2].view(1, 8),
            k_descale=draft.scales[2, :8].view(1, 8),
            v_descale=draft.scales[2, 8:].view(1, 8),
        )

    def forward():
        draft.operator.copy_kv_to_mem_manager(8, slots, kv)
        query = q
        if quant:
            query = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).reshape(2, 32, 128)
        return flash_attn_with_kvcache(q=query, k_cache=k.view(12, 1, 8, 128), v_cache=v.view(12, 1, 8, 128), **kwargs)

    # Warm both the copy kernel and FA3 before capture.
    forward()
    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    draft.kv_buffer.zero_()
    graph.replay()
    query = q
    if quant:
        query = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).reshape(2, 32, 128)
    contiguous_output = flash_attn_with_kvcache(
        q=query, k_cache=k.contiguous().view(12, 1, 8, 128), v_cache=v.contiguous().view(12, 1, 8, 128), **kwargs
    )
    torch.testing.assert_close(output, contiguous_output, atol=0, rtol=0)
    if quant:
        q_ref = q_per_head_static_fp8_quant(q.reshape(2, 8, -1), draft.q_scales[2]).float()
        q_ref = (q_ref * draft.q_scales[2].view(1, 8, 1)).reshape(2, 32, 128)
        k_ref = k[:4].float() * draft.scales[2, :8].view(1, 8, 1)
        v_ref = v[:4].float() * draft.scales[2, 8:].view(1, 8, 1)
    else:
        q_ref, k_ref, v_ref = q.float(), k[:4].float(), v[:4].float()
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q_ref.transpose(0, 1).unsqueeze(0),
            k_ref.repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0),
            v_ref.repeat_interleave(4, dim=1).transpose(0, 1).unsqueeze(0),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    # FP8 FA3 also rounds intermediate attention probabilities.
    tolerance = 0.06 if quant else 0.015
    torch.testing.assert_close(output.float(), expected, atol=tolerance, rtol=tolerance)
    assert torch.count_nonzero(target.kv_buffer[:2]) == 0
