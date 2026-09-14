import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.core.objs.start_args_type import StartArgs


VOCAB_TOPK_CHOICES = [16, 32, 64, 128, 256, 512]


def test_vocab_topk_sampling_defaults_are_disabled():
    cli_args = make_argument_parser().parse_args([])
    start_args = StartArgs()

    assert cli_args.target_vocab_topk_sampling is None
    assert cli_args.draft_vocab_topk_sampling is None
    assert start_args.target_vocab_topk_sampling is None
    assert start_args.draft_vocab_topk_sampling is None


@pytest.mark.parametrize("top_k", VOCAB_TOPK_CHOICES)
def test_target_and_draft_vocab_topk_sampling_accept_supported_widths(top_k):
    args = make_argument_parser().parse_args(
        ["--target_vocab_topk_sampling", str(top_k), "--draft_vocab_topk_sampling", str(top_k)]
    )

    assert args.target_vocab_topk_sampling == top_k
    assert args.draft_vocab_topk_sampling == top_k


@pytest.mark.parametrize("option", ["--target_vocab_topk_sampling", "--draft_vocab_topk_sampling"])
def test_vocab_topk_sampling_rejects_unsupported_width(option):
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args([option, "1"])


def test_removed_vocab_parallel_sampling_option_is_rejected():
    with pytest.raises(SystemExit):
        make_argument_parser().parse_args(["--vocab_parallel_sampling", "draft"])
