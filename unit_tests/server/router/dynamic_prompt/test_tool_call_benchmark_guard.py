import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[4] / "test/benchmark/service/tool_call_cache/run_paired.py"
spec = importlib.util.spec_from_file_location("tool_call_paired_guard", path)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


def sample(index="0", active="Not Active", software=100, hardware=200):
    return (
        f"{index}, 100 %, 300 W, 40000 MiB, 80, 1980 MHz, 3201 MHz, {active}, Not Active, {software} us, {hardware} us"
    )


@pytest.mark.parametrize(
    "active,software,expected", [("Active", 100, True), ("Not Active", 101, True), ("Not Active", 100, False)]
)
def test_detects_active_and_between_sample_thermal_slowdown(active, software, expected):
    baseline = {}
    assert not paired.thermal_events([sample()], {"0"}, baseline)
    events = paired.thermal_events([sample(active=active, software=software)], {"0"}, baseline)
    assert bool(events) is expected


def test_unrelated_gpu_thermal_activity_does_not_disqualify_exclusive_trial_gpus():
    assert not paired.thermal_events([sample(index="4", active="Active")], {"0", "1"}, {})


def test_counter_reset_invalidates_hardware_observation():
    baseline = {}
    paired.thermal_events([sample()], {"0"}, baseline)
    with pytest.raises(ValueError, match="counters reset"):
        paired.thermal_events([sample(software=0)], {"0"}, baseline)
