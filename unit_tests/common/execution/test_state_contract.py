from types import SimpleNamespace

import pytest

from lightllm.common.execution.contracts import FormatSpec, LayerStates, StateSpec, StateView


class ValidParams:
    def validate(self):
        pass


def make_state(*, consumers=(0, 1)):
    return StateSpec("history", "kv", 0, consumers, ValidParams())


def test_state_spec_exposes_stable_identity_and_shared_consumers():
    state = make_state()

    assert state.key == ("history", "kv")
    assert state.producer == 0
    assert state.consumers == (0, 1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"role": "", "resource": "kv"}, "identity"),
        ({"consumers": ()}, "identity"),
        ({"producer": 2}, "include the producer"),
        ({"consumers": (0, 0)}, "unique"),
        ({"consumers": (-1,)}, "nonnegative"),
    ],
)
def test_state_spec_rejects_invalid_ownership(kwargs, message):
    values = {"role": "history", "resource": "kv", "producer": 0, "consumers": (0,), "params": ValidParams()}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        StateSpec(**values)


@pytest.mark.parametrize("field, value", [("producer", 0.5), ("producer", True), ("consumers", [0, 1])])
def test_state_spec_normalizes_or_rejects_layer_id_inputs(field, value):
    values = {"role": "history", "resource": "kv", "producer": 0, "consumers": (0,), "params": ValidParams()}
    values[field] = value

    if field == "consumers" and value == [0, 1]:
        state = StateSpec(**values)
        assert state.consumers == (0, 1)
    else:
        with pytest.raises(ValueError, match="integer"):
            StateSpec(**values)


def test_state_spec_copies_mutable_consumer_inputs():
    consumers = [0, 1]
    state = StateSpec("history", "kv", 0, consumers, ValidParams())

    consumers.append(2)

    assert state.consumers == (0, 1)


def test_format_spec_validates_sharding_and_fields():
    fmt = FormatSpec("plain", object, "paged", ("key", "value"), 16, shard=(1, 2))

    assert fmt.shard == (1, 2)

    with pytest.raises(ValueError, match="group size"):
        FormatSpec("int8", object, "paged", ("value",), 16, group_size=3)
    with pytest.raises(ValueError, match="field dtypes"):
        FormatSpec("plain", object, "paged", ("key", "value"), 16, field_dtypes=(object,))
    with pytest.raises(ValueError, match="integer"):
        FormatSpec("plain", object, "paged", ("value",), 1.5)


def test_state_view_matches_physical_fields_and_layer_state_groups():
    state = make_state(consumers=(0,))
    fmt = FormatSpec("plain", object, "paged", ("key", "value"), 16)
    view = StateView(state, fmt, (SimpleNamespace(), SimpleNamespace()))
    layers = LayerStates(state, auxiliary=(("index", state),))

    assert view.state is state
    assert layers.auxiliary[0][0] == "index"

    with pytest.raises(ValueError, match="view fields"):
        StateView(state, fmt, (object(),))


def test_layer_states_reject_duplicate_auxiliary_names_and_resources():
    state = make_state(consumers=(0,))

    with pytest.raises(ValueError, match="names"):
        LayerStates(state, (("index", state), ("index", state)))
