import importlib


def test_dead_kernel_removed():
    mod = importlib.import_module("lightllm.common.basemodel.triton_kernel.mtp_utils")
    assert not hasattr(mod, "gen_b_req_mtp_start_loc"), "dead gen_b_req_mtp_start_loc must be removed (#22)"
    assert not hasattr(mod, "_fwd_kernel_gen_b_req_mtp_start_loc"), "dead kernel must be removed (#22)"
