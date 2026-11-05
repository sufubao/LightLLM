from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM
from .mxfp4_impl import FuseMoeMXFP4


def select_fuse_moe_impl(quant_method: QuantizationMethod, enable_ep_moe: bool):
    if enable_ep_moe:
        return FuseMoeDeepGEMM

    if quant_method.method_name == "awq_marlin":
        return FuseMoeMarlin
    elif quant_method.method_name == "mxfp4":
        return FuseMoeMXFP4
    else:
        return FuseMoeTriton
