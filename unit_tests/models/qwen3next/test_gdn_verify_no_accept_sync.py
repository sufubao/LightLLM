import pathlib

SRC = (
    pathlib.Path(__file__).resolve().parents[3]
    / "lightllm/models/qwen3next/layer_infer/transformer_layer_infer.py"
)


def test_no_per_step_accept_all_sync():
    text = SRC.read_text()
    assert "infer_state.b_num_accepted_tokens >= 1).all()" not in text, (
        "_gdn_verify_kernel still runs a per-step .all() D2H sync on b_num_accepted_tokens; "
        "the bound is guaranteed upstream (#8b)."
    )
    # the cheap structural assert must remain
    assert "b_ssm_index_rows.dim() == 2" in text, "keep the no-sync structural assert"
