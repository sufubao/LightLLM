import copy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/replay.py"
spec = importlib.util.spec_from_file_location("tool_call_replay_mix", path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.parametrize("sessions", [4, 8, 16, 24])
@pytest.mark.parametrize("mix,divisor", [("agent25", 4), ("agent50", 2)])
def test_agent_mixtures_have_exact_stable_session_counts(sessions, mix, divisor):
    selected = [(f"s{i}", [{"turn": 0}], 0) for i in range(sessions)]
    kinds = replay.stream_kinds(selected, mix)
    assert len(kinds) == sessions
    assert list(kinds.values()).count("agent") == sessions // divisor
    assert list(kinds.values()).count("one-shot") == sessions - sessions // divisor
    assert kinds == replay.stream_kinds(copy.deepcopy(selected), mix)


@pytest.mark.parametrize("mix", ["trace", "ordinary", "all-tools", "mixed25"])
def test_existing_modes_keep_their_original_stream_semantics(mix):
    selected = [("a", [], 0), ("b", [], 0)]
    assert replay.stream_kinds(selected, mix) == {"a": "trace", "b": "trace"}


def test_independent_requests_do_not_reuse_a_long_previous_conversation():
    system = {"role": "system", "content": "Session s. Follow instructions."}
    original = copy.deepcopy(system)
    messages = [system, {"role": "user", "content": "OLD_PRIVATE_REFERENCE" * 1000}]
    replay.reset_one_shot(messages, system, 0)
    first = copy.deepcopy(messages)
    messages.extend([{"role": "assistant", "content": "OK"}, {"role": "user", "content": "MORE_OLD_DATA"}])
    replay.reset_one_shot(messages, system, 1)
    assert len(messages) == 1
    assert messages != first
    assert "OLD" not in str(messages)
    assert system == original


def test_cohort_latency_statistics_keep_empty_groups_explicit():
    assert replay.latency_stats([]) == {"mean": None, "p50": None, "p95": None}
    result = replay.latency_stats([1, 2, 3, 4])
    assert result["mean"] == 2.5
    assert result["p95"] >= result["p50"]
