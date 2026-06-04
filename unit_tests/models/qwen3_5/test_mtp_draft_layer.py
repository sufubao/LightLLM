import os
import pytest
import torch

CKPT = os.environ.get("QWEN35_MTP_CKPT", "/mtc/models/Qwen3.5-27B")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(CKPT),
    reason="needs CUDA + a qwen3_5 checkpoint with mtp.* weights",
)


def test_draft_single_layer_is_full_attention_with_mrope():
    """Risk #12: a naive inherit gives a GDN layer or standard rope. The draft's
    one layer must take the full-attn (mrope) path, NOT a GDN path. Full logits
    parity is covered E2E in Phase 10; this marks the contract."""
    pytest.skip("Implement with checkpoint fixture; logits parity covered E2E in Phase 10.")
