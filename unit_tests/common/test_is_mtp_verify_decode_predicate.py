def test_predicate():
    from lightllm.common.basemodel.batch_objs import is_mtp_verify_decode

    sentinel = object()
    assert is_mtp_verify_decode(3, sentinel) is True
    assert is_mtp_verify_decode(3, None) is False
    assert is_mtp_verify_decode(0, sentinel) is False
    assert is_mtp_verify_decode(0, None) is False
