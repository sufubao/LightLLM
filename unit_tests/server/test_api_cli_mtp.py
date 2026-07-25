from lightllm.server.api_cli import make_argument_parser


def test_max_mtp_step_is_the_public_option():
    args = make_argument_parser().parse_args(["--max_mtp_step", "4"])

    assert args.max_mtp_step == 4
    assert not hasattr(args, "mtp_step")


def test_legacy_mtp_step_remains_parse_compatible():
    args = make_argument_parser().parse_args(["--mtp_step", "3"])

    assert args.max_mtp_step == 0
    assert args.mtp_step == 3


def test_mtp_scheduler_profile_is_public_option():
    args = make_argument_parser().parse_args(["--mtp_scheduler_profile", "profile.json"])

    assert args.mtp_scheduler_profile == "profile.json"
