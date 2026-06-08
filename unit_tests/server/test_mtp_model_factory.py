import pytest


def test_wrong_mtp_mode_rejected_before_construction():
    from lightllm.server.router.model_infer.mode_backend.mtp_model_factory import create_mtp_draft_model

    # deepseek_v3 only supports *_with_att; a *_no_att mode must assert before constructing.
    with pytest.raises(AssertionError):
        create_mtp_draft_model("deepseek_v3", "vanilla_no_att", {})


def test_unknown_model_type_raises_valueerror():
    from lightllm.server.router.model_infer.mode_backend.mtp_model_factory import create_mtp_draft_model

    with pytest.raises(ValueError, match="Unsupported MTP model type"):
        create_mtp_draft_model("not_a_real_model_type", "vanilla_with_att", {})
