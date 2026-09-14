"""An experiment runs its trials, records every stage, and never repeats one."""

import json

import pytest

from strata.experiment import ExperimentError, Ledger, load, run_experiment
from strata.experiment.run import Handles
from strata.modelling import RunStore


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
        assert result.metrics["exact_match"] > 0.5
        assert result.metrics["exact_match"] == result.metrics["recall"] == result.metrics["f1"]
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


def test_a_grid_varies_the_project_as_configured(project, catalog, experiment_file, tmp_path):
    # The project's own parameters are the base; the file's and the grid's
    # land on top, key by key
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[model.params]\n", "[model.params]\nepochs = 3\n"))
    from strata.labeller import Project

    handles = _handles(Project.load(project.root), catalog, tmp_path)
    [a, b] = run_experiment(load(experiment_file), handles)
    assert a.records[3].request["params"] == {"epochs": 3, "lr": 0.1}
    assert b.records[3].request["params"] == {"epochs": 3, "lr": 0.5}


def test_an_unlocked_split_is_a_drawn_one(project, catalog, experiment_file, tmp_path):
    handles = _handles(project, catalog, tmp_path)
    drawn = load(experiment_file, {"split.seed": 7, "split.holdout_ratio": 0.2})
    [a, b] = run_experiment(drawn, handles)
    assert a.records[2].record["drawn"] is True
    assert a.records[2].record["directory"] != a.records[1].record["directory"]
    # The draw itself is what each run saw, in the run store: both trials
    # trained on the same drawn sides, and the holdout is a real one
    store = RunStore.local(project.runs_dir)
    assert store.saw(a.run_id) == store.saw(b.run_id)
    assert store.saw(a.run_id)["holdout"]


def test_a_run_names_its_experiment_and_what_it_saw(project, catalog, experiment_file, tmp_path):
    experiment = load(experiment_file)
    [a, b] = run_experiment(experiment, _handles(project, catalog, tmp_path))
    store = RunStore.local(project.runs_dir)
    assert store.get(a.run_id).experiment_id == experiment.id
    # A study is a query over the store
    assert [r.id for r in store.for_experiment(experiment.id)] == [a.run_id, b.run_id]
    saw = store.saw(a.run_id)
    assert {side: len(sums) for side, sums in saw.items()} == a.records[2].record["counts"]


def test_a_change_in_the_project_reruns_what_reads_it(project, catalog, experiment_file, tmp_path):
    handles = _handles(project, catalog, tmp_path)
    run_experiment(load(experiment_file), handles)
    # The file is unchanged; the project it runs over is not
    toml = project.root / "project.toml"
    toml.write_text(toml.read_text().replace("[model.params]\n", "[model.params]\nepochs = 3\n"))
    from strata.labeller import Project

    handles = _handles(Project.load(project.root), catalog, tmp_path)
    [a, b] = run_experiment(load(experiment_file), handles)
    assert [r.reused for r in a.records] == [True, True, True, False, False]
    assert a.records[3].request["params"] == {"epochs": 3, "lr": 0.1}


def test_the_ledger_holds_no_path_of_this_machine(project, catalog, experiment_file, tmp_path):
    [a, _] = run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))
    for record in a.records:
        for value in record.request.values():
            assert not str(value).startswith(str(project.root)), (record.stage, value)
    assert a.records[3].request["dataset_dir"].startswith("datasets/")
    assert a.records[3].request["model"] == "toy.py:Toy"


def test_a_file_loaded_for_another_project_is_refused(project, catalog, experiment_file, tmp_path):
    experiment = load(experiment_file).model_copy(update={"project_id": "other"})
    with pytest.raises(ExperimentError, match="'other'"):
        run_experiment(experiment, _handles(project, catalog, tmp_path))
