from types import SimpleNamespace

import pytest
import torch

from lightllm.common.req_manager.linear_att import ReqManagerForMamba


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_capture_and_private_restore_of_committed_mtp_state(offset):
    manager = ReqManagerForMamba.__new__(ReqManagerForMamba)
    manager.mtp_step = 3
    manager.linear_config = SimpleNamespace(conv_kernel_size=4)
    manager.req_to_mtp_state_index = torch.tensor([offset, 3], dtype=torch.int32)
    manager.req_to_conv_state = SimpleNamespace(buffer=torch.arange(2 * 2 * 4 * 6).reshape(2, 2, 4, 6).float())
    manager.req_to_ssm_state = SimpleNamespace(buffer=torch.arange(2 * 8 * 4 * 4).reshape(2, 8, 4, 4).float())
    source = SimpleNamespace(req_idx=0)
    destination = SimpleNamespace(req_idx=1)
    conv, ssm = manager.get_linear_att_state(source)
    assert torch.equal(conv, manager.req_to_conv_state.buffer[:, 0, :, offset : offset + 3])
    assert torch.equal(ssm, manager.req_to_ssm_state.buffer[:, offset])
    saved_conv, saved_ssm = conv.clone(), ssm.clone()
    manager.restore_linear_att_state(destination, saved_conv, saved_ssm)
    assert manager.req_to_mtp_state_index[1] == 0
    assert torch.equal(manager.req_to_conv_state.buffer[:, 1, :, :3], saved_conv)
    assert torch.equal(manager.req_to_ssm_state.buffer[:, 4], saved_ssm)
    manager.req_to_ssm_state.buffer[:, 4].add_(100)
    assert torch.equal(saved_ssm, ssm)


def test_output_capture_excludes_uncomputed_token_and_mtp_stop_overrun():
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    captured = []
    context = InferenceContext()
    context.is_linear_att_mixed_model = True
    context.req_manager = SimpleNamespace(req_to_mtp_state_index=torch.tensor([3]))
    context.backend = SimpleNamespace(
        linear_att_checkpoint_cache=SimpleNamespace(
            capture=lambda manager, req, length, state_index: captured.append((length, state_index))
        )
    )
    # Four tokens verified. Stop at the second emitted token; cache the state
    # immediately before it, whose recurrent offset is 1, not the final 3.
    req = SimpleNamespace(
        req_idx=0, cur_kv_len=103, mtp_step=3, linear_output_captured=False, linear_output_cache_len=101
    )
    context.capture_output_linear_state(req)
    context.capture_output_linear_state(req)
    assert captured == [(101, 1)]


def test_interval_checkpoint_inside_accepted_output_bundle():
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    captured = []
    context = InferenceContext()
    context.is_linear_att_mixed_model = True
    context.req_manager = SimpleNamespace(req_to_mtp_state_index=torch.tensor([3]))
    context.backend = SimpleNamespace(
        linear_att_checkpoint_cache=SimpleNamespace(
            policy=SimpleNamespace(interval=1024),
            capture=lambda manager, req, length, state_index: captured.append((length, state_index)),
        )
    )
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=1026,
        mtp_step=3,
        shm_req=SimpleNamespace(input_len=1000),
        linear_output_cache_len=None,
        linear_last_output_checkpoint=0,
    )
    context.capture_decode_linear_states([req, req, req])
    context.capture_decode_linear_states([req])
    assert captured == [(1024, 1)]


def test_nonchunked_prefill_labels_state_with_actual_full_length():
    from lightllm.common.linear_att_cache_manager.checkpoints import CheckpointPolicy
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    captured = []
    context = InferenceContext()
    context.is_linear_att_mixed_model = True
    context.backend = SimpleNamespace(
        disable_chunked_prefill=True,
        linear_att_checkpoint_cache=SimpleNamespace(
            policy=CheckpointPolicy(interval=1024, hash_page_size=64),
            capture=lambda manager, req, length, state_index: captured.append((length, state_index)),
        ),
    )
    req = SimpleNamespace(
        shm_req=SimpleNamespace(input_len=1103),
        linear_checkpoint_demand=0,
        get_cur_total_len=lambda: 1103,
        get_chuncked_input_token_len=lambda: 256,
    )
    context.copy_linear_att_state_to_cache_buffer(None, [req])
    assert captured == [(1103, 0)]
