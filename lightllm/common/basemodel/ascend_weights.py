from functools import partial

import torch


def embedding(input_ids, out=None, alloc_func=torch.empty, *, weight, start, end):
    valid = (input_ids >= start) & (input_ids < end)
    indices = (input_ids - start).clamp(0, weight.shape[0] - 1)
    result = torch.nn.functional.embedding(indices, weight) * valid[:, None]
    if out is not None:
        out.copy_(result)
        return out
    return result


def lm_head(input, out=None, alloc_func=torch.empty, *, weight):
    if out is None:
        out = alloc_func((weight.shape[0], input.shape[1]), dtype=weight.dtype, device=weight.device)
    return torch.mm(weight, input, out=out)


def rms_norm(input, eps, out=None, alloc_func=torch.empty, *, weight):
    import torch_npu

    result = torch_npu.npu_rms_norm(input, weight, epsilon=eps)[0]
    if out is not None:
        out.copy_(result)
        return out
    return result


def finalize_weights(layers, runtime):
    from lightllm.common.basemodel.layer_weights.meta_weights.embedding_weight import EmbeddingWeight, LMHeadWeight
    from lightllm.common.basemodel.layer_weights.meta_weights.norm_weight import RMSNormWeight

    for layer in layers:
        for weight in vars(layer).values():
            if type(weight) is LMHeadWeight:
                weight._forward = partial(lm_head, weight=weight.weight)
            elif type(weight) is EmbeddingWeight:
                weight._forward = partial(
                    embedding, weight=weight.weight, start=weight.tp_vocab_start_id, end=weight.tp_vocab_end_id
                )
            elif type(weight) is RMSNormWeight:
                weight._forward = partial(rms_norm, weight=weight.weight)
