import types


def test_pure_helper_mapping():
    from lightllm.utils.envs_utils import _mtp_added_layer_num

    assert _mtp_added_layer_num("eagle_with_att", 3) == 1
    assert _mtp_added_layer_num("vanilla_with_att", 3) == 3
    assert _mtp_added_layer_num("vanilla_no_att", 3) == 0
    assert _mtp_added_layer_num("eagle_no_att", 3) == 0
    assert _mtp_added_layer_num(None, 3) == 0


def test_config_objs_delegates_to_helper():
    from lightllm.utils.envs_utils import _mtp_added_layer_num
    from lightllm.common.linear_att_cache_manager.config_objs import get_mtp_draft_full_att_layer_num

    for mode in ("eagle_with_att", "vanilla_with_att", "vanilla_no_att", None):
        args = types.SimpleNamespace(mtp_mode=mode, mtp_step=3)
        assert get_mtp_draft_full_att_layer_num(args) == _mtp_added_layer_num(mode, 3)
