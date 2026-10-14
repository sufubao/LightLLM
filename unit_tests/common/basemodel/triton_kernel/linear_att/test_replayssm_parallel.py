import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("capacity", [8, 16, 64])
def test_derived_fold_commits_only_accepted_history(capacity):
    torch.manual_seed(731)
    state = torch.randn(2, 5, 4, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.02
    initial = state.clone()
    cache = ReplaySSMCache(state, capacity, 5, num_key_heads=2, projection_mode="parallel", fold_method="derived")
    cache.keys.fill_(float("nan"))
    cache.deltas.fill_(float("nan"))
    cache.gates.fill_(float("nan"))
    cache.raw_keys.fill_(float("nan"))
    cache.raw_values.fill_(float("nan"))
    cache.betas.fill_(float("nan"))
    # Fold request 2 from the second ring half; request 0 has an empty span.
    count = capacity - 2
    cache.cursors[2] = 2 * capacity + count
    cache.cursors[0] = count
    cache.keys[:, 2, :, capacity : capacity + count].normal_(0, 0.1)
    cache.deltas[:, 2, :, capacity : capacity + count].normal_(0, 0.1)
    cache.gates[:, 2, :, capacity : capacity + count].fill_(-0.1)
    expected = initial[:, 2].float()
    for j in range(count):
        keys = cache.keys[:, 2, :, capacity + j].repeat_interleave(2, dim=1)
        delta = cache.deltas[:, 2, :, capacity + j]
        decay = cache.gates[:, 2, :, capacity + j].exp()
        expected = expected * decay[:, :, None, None] + keys[:, :, :, None] * delta[:, :, None, :]
    reqs = torch.tensor([2, 0, 4], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 5, 5, 10], dtype=torch.int32, device="cuda")
    cache.prepare_decode(reqs, cu)
    assert torch.isfinite(state).all()
    torch.testing.assert_close(state[:, 2], expected.bfloat16(), atol=2e-4, rtol=0.008)
    torch.testing.assert_close(state[:, [0, 1, 3, 4]], initial[:, [0, 1, 3, 4]], atol=0, rtol=0)
    assert cache.cursors[2] == 0
    assert cache.cursors[0] == count
    assert cache.cursors[4] == 0


@pytest.mark.parametrize("fold", ["raw", "derived"])
def test_parallel_rejects_unsupported_shapes(fold):
    state = torch.zeros(1, 3, 4, 32, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="128x128"):
        ReplaySSMCache(state, 16, 5, num_key_heads=2, projection_mode="parallel", fold_method=fold)


def test_derived_fold_requires_parallel_solver():
    state = torch.zeros(1, 3, 4, 128, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        ReplaySSMCache(state, 16, 5, num_key_heads=2, fold_method="derived")


@pytest.mark.parametrize("width", [2, 3, 5, 8, 16])
def test_parallel_raw_against_sequential_reference(width):
    from unit_tests.common.basemodel.triton_kernel.linear_att.test_replayssm import (
        test_replay_acceptance_flush_and_graph as check_replay,
    )

    check_replay(
        width=width,
        dims=(128, 128),
        capacity=16,
        dtype=torch.bfloat16,
        projection_mode="parallel",
        run_config={"BV": 128, "num_warps": 2, "num_stages": 1},
    )
