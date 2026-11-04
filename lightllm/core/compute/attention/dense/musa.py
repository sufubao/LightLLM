from lightllm.core.contracts.provider import ProbeResult
from .backend import DenseAttentionBackend


class MusaAttentionBackend(DenseAttentionBackend):
    def __init__(self, phase):
        super().__init__("triton", phase)

    def supports_device(self, caps):
        return caps.device_type == "musa" and caps.platform == "musa"

    def probe(self, spec, format, caps, limits):
        if self.phase == "verify" or limits.dynamic_verify or limits.max_verify_tokens != 1:
            return ProbeResult(reason="MUSA Dense attention does not support speculative verification")
        if limits.transfer_paths or limits.expert_parallel or limits.microbatch_count != 1:
            return ProbeResult(reason="MUSA Dense attention requires local state without EP or microbatch overlap")
        if any(phase != "decode" for phase in limits.graph_phases):
            return ProbeResult(reason="MUSA currently supports decode Graph capture only")
        return super().probe(spec, format, caps, limits)
