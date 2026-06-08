import torch


def _listcomp_start_loc(b_mtp_index):
    return [i for i, m in enumerate(b_mtp_index) if m == 0]


def test_arange_equals_listcomp_for_contiguous_expanded_batch():
    for S in (0, 1, 2, 3):
        for n_real in (1, 4, 7):
            b_mtp_index = torch.arange(S + 1, dtype=torch.int32).repeat(n_real)  # 0..S, 0..S, ...
            expected = _listcomp_start_loc(b_mtp_index.tolist())
            got = (torch.arange(n_real, dtype=torch.int32) * (S + 1)).tolist()
            assert got == expected, f"S={S} n_real={n_real}: {got} != {expected}"
