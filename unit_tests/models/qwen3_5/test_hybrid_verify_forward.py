import os
import pytest
import torch

CKPT = os.environ.get("QWEN35_MTP_CKPT", "/mtc/models/Qwen3.5-27B")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(CKPT),
    reason="needs CUDA + a qwen3_5 checkpoint (QWEN35_MTP_CKPT or /mtc/models/Qwen3.5-27B)",
)


def test_hybrid_mtp_verify_matches_sequential_decode():
    """A verify step over S+1 fully-accepted candidates must produce the same
    committed hidden state / next-token logits as sequentially decoding those
    tokens through the non-MTP path, across BOTH GDN and full-attn layers.
    Full end-to-end equivalence is enforced E2E in Phase 10; this scaffold marks
    the per-layer-dispatch contract (design §3.4b)."""
    pytest.skip("Implement with the running-model fixture; covered E2E in Phase 10.")
