"""An experiment runs its trials, records every stage, and never repeats one."""

import json

from strata.experiment import Ledger, load, run_experiment
from strata.experiment.run import Handles


def _handles(project, catalog, tmp_path) -> Handles:
    return Handles(project=project, catalog=catalog, catalog_root=tmp_path / "catalog")


def test_a_grid_runs_every_trial_and_scores_each_on_the_holdout(
    project, catalog, experiment_file, tmp_path
):
    results = run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))

    assert [r.trial.overrides for r in results] == [
        {"train.params.lr": 0.1}, {"train.params.lr": 0.5}
    ]
    for result in results:
        assert [r.stage for r in result.records] == [
            "dataset", "materialise", "split", "train", "evaluate"
        ]
        assert result.run_id is not None
        # The toy predicts cat for everything and two in three answers are cat,
        # so every micro figure is the share of cats on the holdout
        assert result.metrics["accuracy"] > 0.5
        assert result.metrics["accuracy"] == result.metrics["recall"] == result.metrics["f1"]
    assert results[0].run_id != results[1].run_id


def test_the_stages_the_trials_share_ran_once(project, catalog, experiment_file, tmp_path):
    results = run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))
    first, second = results
    # dataset, materialise and split are identical through the prefix, so
    # the second trial found them; train and evaluate it had to run
    assert [r.reused for r in first.records] == [False] * 5
    assert [r.reused for r in second.records] == [True, True, True, False, False]
    assert first.records[0].record["dataset_id"] == second.records[0].record["dataset_id"]


def test_a_second_run_does_nothing(project, catalog, experiment_file, tmp_path):
    handles = _handles(project, catalog, tmp_path)
    first = run_experiment(load(experiment_file), handles)
    again = run_experiment(load(experiment_file), handles)
    assert all(r.reused == 5 for r in again)
    assert [r.run_id for r in again] == [r.run_id for r in first]


def test_a_changed_argument_reruns_only_what_is_downstream(
    project, catalog, experiment_file, tmp_path
):
    handles = _handles(project, catalog, tmp_path)
    run_experiment(load(experiment_file), handles)
    tuned = load(experiment_file, {"train.params.lr": 0.7})
    [result] = run_experiment(tuned, handles)
    assert [r.reused for r in result.records] == [True, True, True, False, False]


def test_the_ledger_is_what_a_person_can_read(project, catalog, experiment_file, tmp_path):
    experiment = load(experiment_file)
    run_experiment(experiment, _handles(project, catalog, tmp_path))
    root = project.root / "experiments" / experiment.short_id
    assert json.loads((root / "experiment.json").read_text())["name"] == "lr"
    # Every record by key, shared by whichever trial or experiment made it
    assert len(list((project.root / "experiments" / "records").glob("*.json"))) == 7
    trial_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(trial_dirs) == 2
    names = sorted(p.name for p in trial_dirs[0].iterdir())
    assert names == [
        "01-dataset.json",
        "02-materialise.json",
        "03-split.json",
        "04-train.json",
        "05-evaluate.json",
        "trial.json",
    ]
    train = json.loads((trial_dirs[0] / "04-train.json").read_text())
    assert train["request"]["params"] in ({"lr": 0.1}, {"lr": 0.5})
    assert train["record"]["run_id"]


def test_the_ledger_can_be_placed_elsewhere(project, catalog, experiment_file, tmp_path):
    ledger = Ledger(tmp_path / "ledger")
    run_experiment(load(experiment_file), _handles(project, catalog, tmp_path), ledger=ledger)
    assert (tmp_path / "ledger").exists()
    assert not (project.root / "experiments").exists()


def test_an_unlocked_split_is_a_drawn_one(project, catalog, experiment_file, tmp_path):
    handles = _handles(project, catalog, tmp_path)
    drawn = load(experiment_file, {"split.seed": 7, "split.holdout_ratio": 0.2})
    [a, b] = run_experiment(drawn, handles)
    assert a.records[2].record["drawn"] is True
    assert a.records[2].record["directory"] != a.records[1].record["directory"]
    assert a.records[2].record["sides"] == b.records[2].record["sides"]
