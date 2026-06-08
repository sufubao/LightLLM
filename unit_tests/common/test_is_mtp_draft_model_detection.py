import types


def test_detection_uses_attribute_not_string():
    from lightllm.common.basemodel.cuda_graph import CudaGraph
    from lightllm.models.base_mtp_model import BaseMTPModel

    graph = CudaGraph.__new__(CudaGraph)

    class _Draft(BaseMTPModel):
        pass

    class _Main:
        pass

    assert graph._is_mtp_draft_model(_Draft.__new__(_Draft)) is True
    assert graph._is_mtp_draft_model(_Main()) is False
    # a non-MTP class whose name happens to contain "MTPModel" must NOT be misdetected
    GotchaMTPModelButNot = types.new_class("GotchaMTPModelButNot")
    assert graph._is_mtp_draft_model(GotchaMTPModelButNot()) is False
