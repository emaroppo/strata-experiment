"""The experiment file: what it refuses, what it hashes, and how a grid expands."""

import pytest

from strata.experiment import ChainError, ExperimentError, load
from strata.experiment.registry import check
from strata.experiment.spec import from_payload


def _payload(**overrides):
    payload = {
        "project": "demo",
        "name": "x",
        "stage": [
            {"use": "dataset", "holdout_ratio": 0.1},
            {"use": "materialise"},
            {"use": "split"},
            {"use": "train", "fresh": True, "params": {"lr": 0.1}},
            {"use": "evaluate", "on": "holdout"},
        ],
    }
    payload.update(overrides)
    return payload


def test_a_file_loads_and_its_stages_keep_their_arguments(experiment_file):
    experiment = load(experiment_file)
    assert [s.use for s in experiment.stages] == [
        "dataset", "materialise", "split", "train", "evaluate"
    ]
    assert experiment.stages[0].args == {"val_ratio": 0.2, "holdout_ratio": 0.2}
    assert experiment.stages[3].args == {"fresh": True, "params": {"lr": 0.1}}


def test_the_id_is_over_meaning_not_text():
    a = from_payload(_payload())
    b = from_payload(_payload())
    b_payload = _payload()
    b_payload["stage"][3] = {"params": {"lr": 0.1}, "fresh": True, "use": "train"}
    assert a.id == b.id == from_payload(b_payload).id
    changed = _payload()
    changed["stage"][3]["params"]["lr"] = 0.2
    assert from_payload(changed).id != a.id


def test_a_grid_expands_to_every_combination_in_order():
    experiment = from_payload(
        _payload(grid={"train.params.lr": [0.1, 0.5], "dataset.seed": [1, 2]})
    )
    trials = experiment.trials()
    assert [t.overrides for t in trials] == [
        {"train.params.lr": 0.1, "dataset.seed": 1},
        {"train.params.lr": 0.1, "dataset.seed": 2},
        {"train.params.lr": 0.5, "dataset.seed": 1},
        {"train.params.lr": 0.5, "dataset.seed": 2},
    ]
    assert trials[1].experiment.stages[0].args["seed"] == 2
    assert trials[2].experiment.stages[3].args["params"] == {"lr": 0.5}
    assert not trials[0].experiment.grid
    assert len({t.id for t in trials}) == 4


def test_no_grid_is_one_trial_as_written():
    [trial] = from_payload(_payload()).trials()
    assert trial.overrides == {} and trial.label == "as written"


def test_a_prefix_hash_moves_only_downstream_of_a_change():
    base = from_payload(_payload())
    changed = base.with_overrides({"train.params.lr": 0.9})
    assert [base.prefix_hash(i) for i in range(3)] == [changed.prefix_hash(i) for i in range(3)]
    assert base.prefix_hash(3) != changed.prefix_hash(3)
    assert base.prefix_hash(4) != changed.prefix_hash(4)


def test_overrides_at_load_are_part_of_the_identity(experiment_file):
    plain = load(experiment_file)
    tuned = load(experiment_file, {"train.params.lr": 0.3})
    assert tuned.id != plain.id
    assert tuned.stages[3].args["params"] == {"lr": 0.3}


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_a_stage_without_a_name_is_refused():
    with pytest.raises(ExperimentError, match="needs `use`"):
        from_payload(_payload(stage=[{"val_ratio": 0.2}]))


def test_an_unknown_key_is_refused():
    with pytest.raises(ExperimentError, match="extra"):
        from_payload(_payload(nmae="typo"))


def test_a_grid_over_a_stage_not_in_the_file_is_refused():
    with pytest.raises(ExperimentError, match="does not use"):
        from_payload(_payload(grid={"push.limit": [1]}))


def test_a_grid_over_a_repeated_stage_is_refused():
    payload = _payload()
    payload["stage"].append({"use": "evaluate", "on": "val"})
    with pytest.raises(ExperimentError, match="ambiguous"):
        from_payload(payload | {"grid": {"evaluate.on": ["val"]}})


def test_an_unknown_stage_is_refused_by_the_chain():
    with pytest.raises(ChainError, match="No stage named 'push'"):
        check(from_payload(_payload(stage=[{"use": "push"}])))


def test_an_argument_a_stage_does_not_take_is_refused():
    with pytest.raises(ChainError, match="does not take val_ration"):
        check(from_payload(_payload(stage=[{"use": "dataset", "val_ration": 0.2}])))


def test_a_stage_before_what_it_needs_is_refused():
    payload = _payload(stage=[{"use": "dataset"}, {"use": "train", "fresh": True}])
    with pytest.raises(ChainError, match="needs a dataset_dir"):
        check(from_payload(payload))


def test_a_grid_needs_every_trial_cold_or_pinned():
    payload = _payload(grid={"train.params.lr": [0.1, 0.2]})
    payload["stage"][3] = {"use": "train", "params": {"lr": 0.1}}
    with pytest.raises(ChainError, match="fresh = true"):
        check(from_payload(payload))
    payload["stage"][3] = {"use": "train", "parent": "r1", "params": {"lr": 0.1}}
    check(from_payload(payload))
