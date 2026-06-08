def test_retarget_swaps_prefix_once():
    from lightllm.models.qwen3_5_mtp.layer_weights.mtp_retarget_mixin import MTPRetargetMixin

    obj = MTPRetargetMixin()
    assert obj._retarget("model.layers.3.self_attn.q_proj.weight") == "mtp.layers.3.self_attn.q_proj.weight"
    assert obj._retarget(None) is None
    assert obj._retarget("model.layers.0.x.model.layers.y") == "mtp.layers.0.x.model.layers.y"


def test_retarget_attn_norm_names_covers_all_attrs():
    from lightllm.models.qwen3_5_mtp.layer_weights.mtp_retarget_mixin import MTPRetargetMixin

    obj = MTPRetargetMixin()
    for attr in MTPRetargetMixin._ATTN_NORM_NAME_ATTRS:
        setattr(obj, attr, "model.layers.5.thing")
    obj._retarget_attn_norm_names()
    for attr in MTPRetargetMixin._ATTN_NORM_NAME_ATTRS:
        assert getattr(obj, attr) == "mtp.layers.5.thing", f"{attr} not retargeted"
