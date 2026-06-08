import torch


def test_accepted_row_idx_selects_one_row_per_request():
    S, n_real = 3, 4
    b_req_mtp_start_loc = torch.arange(n_real, dtype=torch.int32) * (S + 1)  # 0,4,8,12
    mtp_accept_len = torch.tensor([1, 2, 4, 3], dtype=torch.int32)  # in [1, S+1]
    accepted_row_idx = b_req_mtp_start_loc + mtp_accept_len - 1
    assert accepted_row_idx.tolist() == [0, 5, 11, 14]
    for r in range(n_real):
        lo = r * (S + 1)
        assert lo <= accepted_row_idx[r].item() < lo + (S + 1)


def test_mem_index_plan_gather():
    # mirrors chunked _draft_decode_eagle's mem_index_plan = cat([main(S+1), eagle(mtp_step)])
    S, n_real = 2, 3
    mtp_size = S + 1
    main = torch.arange(n_real * mtp_size).view(n_real, mtp_size)  # committed slots
    eagle = (torch.arange(n_real * S) + 100).view(S, n_real).t().contiguous()  # draft slots by req
    plan = torch.cat([main, eagle], dim=1)  # (n_real, mtp_size + S)
    accept = torch.tensor([1, 2, 3], dtype=torch.long)
    accepted_offsets = accept - 1  # 0,1,2
    req = torch.arange(n_real)
    step0 = plan[req, accepted_offsets + 0]
    assert step0.tolist() == [main[0, 0].item(), main[1, 1].item(), main[2, 2].item()]
    step1 = plan[req, accepted_offsets + 1]
    assert step1[2].item() == plan[2, mtp_size].item()


def test_dp_repad_keeps_real_rows_first_and_pads_to_common():
    # DP-specific: shrink to real rows, then re-pad to common_req_num so collectives line up.
    # eagle_mem_indexes is allocated for real reqs only; pad fake reqs with HOLD_TOKEN_MEMINDEX.
    S = 2
    mtp_step = S
    real_req_num = 3
    common_req_num = 5  # this rank padded up to 5 to match peers
    padded_req_num = common_req_num - real_req_num
    HOLD = 999

    mtp_size = mtp_step + 1
    # main mem indexes are laid out (common_req_num, mtp_size) in the padded decode batch
    main_mem_indexes = torch.arange(common_req_num * mtp_size).view(common_req_num, mtp_size)
    # eagle slots allocated as (mtp_step, real_req_num) then transposed to (real_req_num, mtp_step)
    eagle_mem_indexes = torch.arange(mtp_step * real_req_num) + 5000
    eagle_padded = torch.nn.functional.pad(
        eagle_mem_indexes.view(mtp_step, real_req_num).transpose(0, 1).contiguous(),
        (0, 0, 0, padded_req_num),
        value=HOLD,
    )
    assert eagle_padded.shape == (common_req_num, mtp_step)
    # real rows hold the real eagle slots, fake rows are all HOLD
    assert eagle_padded[real_req_num:].eq(HOLD).all()
    assert not eagle_padded[:real_req_num].eq(HOLD).any()

    mem_index_plan = torch.cat([main_mem_indexes, eagle_padded], dim=1)
    assert mem_index_plan.shape == (common_req_num, mtp_size + mtp_step)

    # accepted offsets: real reqs use accept_len-1, fake reqs padded with 0
    mtp_accept_len = torch.tensor([1, 3, 2], dtype=torch.long)
    accepted_offsets = torch.nn.functional.pad(mtp_accept_len - 1, (0, padded_req_num), value=0)
    assert accepted_offsets.tolist() == [0, 2, 1, 0, 0]
    req_offsets = torch.arange(common_req_num, dtype=torch.long)

    # step 0 gather: each real req picks its own committed/eagle slot, fake rows pick a valid main slot
    step0 = mem_index_plan[req_offsets, accepted_offsets + 0]
    assert step0[0].item() == main_mem_indexes[0, 0].item()
    assert step0[1].item() == main_mem_indexes[1, 2].item()  # accept_len 3 -> last main col
    assert step0[2].item() == main_mem_indexes[2, 1].item()
    # fake rows (3,4) at offset 0 select first main column -> never HOLD here (slot unused by real fwd)
    assert step0[3].item() == main_mem_indexes[3, 0].item()

    # step 1 gather: req 1 (offset 2) rolls into first eagle column (a real eagle slot)
    step1 = mem_index_plan[req_offsets, accepted_offsets + 1]
    assert step1[1].item() == eagle_padded[1, 0].item()
    assert step1[1].item() != HOLD
