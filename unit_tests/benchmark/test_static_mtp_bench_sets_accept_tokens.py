import pathlib
import re

BENCH = pathlib.Path(__file__).resolve().parents[2] / "test/benchmark/static_inference/model_infer_mtp.py"


def test_main_decode_sets_b_num_accepted_tokens():
    src = BENCH.read_text()
    # Main verify decode must SET the field (flips is_mtp_verify -> fused GDN verify kernel).
    assert re.search(r"model_input\.b_num_accepted_tokens\s*=\s*torch\.full", src), (
        "static MTP bench no longer sets b_num_accepted_tokens on the main decode ModelInput; "
        "the verify path (production fused GDN kernel) is not being exercised (#5)."
    )


def test_draft_inputs_clear_b_num_accepted_tokens():
    src = BENCH.read_text()
    # Draft forwards must CLEAR it so they take the plain decode layout, mirroring production.
    assert len(re.findall(r"draft_model_input\.b_num_accepted_tokens\s*=\s*None", src)) >= 2, (
        "static MTP bench draft inputs must clear b_num_accepted_tokens (eagle + vanilla)."
    )
