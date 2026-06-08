import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "lightllm/server/router/model_infer/infer_batch.py"


def test_tail_offload_bounds_mtp_accept_len():
    text = SRC.read_text()
    # The tail small-page offload must bound mtp_accept_len before computing canonical_off.
    pattern = re.compile(
        r"assert\s+1\s*<=\s*req\.mtp_accept_len\s*<=\s*self\.args\.mtp_step\s*\+\s*1",
    )
    assert pattern.search(text), (
        "tail small-page conv offload must assert 1 <= req.mtp_accept_len <= mtp_step+1 "
        "before slicing the widened slot at canonical_off (#18)."
    )
