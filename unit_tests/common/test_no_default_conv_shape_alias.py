import pathlib

from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig

BUF = (
    pathlib.Path(__file__).resolve().parents[2]
    / "lightllm/common/linear_att_cache_manager/linear_att_buffer_manager.py"
)


def test_ambiguous_default_alias_removed():
    assert not hasattr(LinearAttCacheConfig, "get_conv_state_shape"), (
        "the default-named get_conv_state_shape() must be removed; callers choose "
        "get_persisted_conv_state_shape() (narrow) or get_gpu_conv_state_shape() (widened) (#24)."
    )
    assert hasattr(LinearAttCacheConfig, "get_persisted_conv_state_shape")
    assert hasattr(LinearAttCacheConfig, "get_gpu_conv_state_shape")


def test_buffer_manager_uses_persisted_shape():
    assert "get_persisted_conv_state_shape()" in BUF.read_text(), (
        "the CPU page buffer must request the persisted (narrow) shape explicitly."
    )
