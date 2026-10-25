"""An experiment scores every checkpoint of a run, and a media model's by what it generates.

The evaluate record says each checkpoint's scores, the ledger keeps what a
generator made beside the records, and the report sets the checkpoints
one under the other. A file that uses none of this keys its stages as it
did before. See ``docs/adr/0054`` and ``docs/adr/0037``.
"""

import json
import re
from pathlib import Path

import pytest

from strata.catalog import Catalog
from strata.contracts import Choices
from strata.experiment import Experiment, StageSpec, load, run_experiment
from strata.experiment.cli import main
from strata.experiment.ledger import stage_key
from strata.experiment.report import headline, ledger_for, rows, scored
from strata.experiment.run import Handles, portable
from strata.modelling.stages import EvaluateRequest, StageError, TrainStageRequest
from strata.project import Project

#: A media model with no framework: it keeps a checkpoint per epoch and
#: writes down which checkpoint was asked about which subject.
ECHO = '''
import json
from pathlib import Path

from strata.contracts import Text
from strata.modelling import Generated, MediaModel


class Echo(MediaModel):
    """Writes "epoch N: <subject>", so an output says what made it."""

    trains_on = Text
    version = "1"

    def __init__(self, epochs: int = 2):
        self.epochs = epochs
        self.epoch = 0

    def finetune(self, train, val=None, on_epoch=None, save_checkpoint=None):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            if save_checkpoint is not None:
                save_checkpoint(f"epoch-{epoch}")
        return {"n_train": float(len(train))}

    def generate(self, inputs, settings, seeds, on_batch=None):
        return [
            Generated(data=f"epoch {self.epoch}: {i.values['subject']}".encode(), ended=True)
            for i in inputs
        ]

    def save(self, path):
        Path(path).write_text(json.dumps({"epoch": self.epoch}))

    def load(self, path):
        self.epoch = json.loads(Path(path).read_text())["epoch"]
'''


@pytest.fixture
def writing(project, tmp_path) -> Project:
    """The project, its model a generator told each message's subject."""
    (project.root / "echo.py").write_text(ECHO)
    toml = project.root / "project.toml"
    text = re.sub(r'^ref = ".*"$', 'ref = "echo.py:Echo"', toml.read_text(), flags=re.M)
    text += '\n[[data.features]]\nname = "subject"\nsource = "metadata"\nref = "subject"\n'
    toml.write_text(text)
    return Project.load(project.root)


@pytest.fixture
def messages(writing, tmp_path) -> Catalog:
    """Fifteen messages, each with a subject, answered so a version can freeze them."""
    catalog = Catalog.local(tmp_path / "catalog")
    raw = tmp_path / "raw"
    raw.mkdir()
    ids = []
    for i in range(15):
        path = raw / f"m{i:02d}.txt"
        path.write_text(f"the body of message {i}")
        ids += catalog.ingest(
            [path],
            media="text",
            metadata={"subject": f"subject {i}"},
            collections=writing.collections,
        )
    label_set = catalog.label_sets.create(writing.label_set_name, writing.label_set.schema)
    catalog.annotations.annotate_many(label_set, [(i, Choices(values=["cat"])) for i in ids])
    return catalog


def _handles(project, catalog, tmp_path) -> Handles:
    return Handles(project=project, catalog=catalog, catalog_root=tmp_path / "catalog")


# ----------------------------------------------------------------------
# A media model, end to end
# ----------------------------------------------------------------------


def test_a_generator_is_scored_at_every_checkpoint_and_its_outputs_kept_beside_the_records(
    writing, messages, tmp_path, write_experiment
):
    experiment = load(write_experiment(writing, "writing"))
    [result] = run_experiment(experiment, _handles(writing, messages, tmp_path))

    evaluated = result.records[-1].record
    assert [c["name"] for c in evaluated["checkpoints"]] == ["epoch-1", "epoch-2"]
    assert set(result.by_checkpoint) == {"epoch-1", "epoch-2"}
    assert all("text-generation.ended" in m for m in result.by_checkpoint.values())
    # What a person reads: each output, by the file the record names
    outputs = writing.root / "experiments" / "outputs"
    for epoch, checkpoint in enumerate(evaluated["checkpoints"], start=1):
        assert checkpoint["outputs"] == []
        lines = [
            json.loads(x)
            for x in (outputs / checkpoint["outputs_file"]).read_text().split("\n")
            if x
        ]
        assert len(lines) == evaluated["samples"]
        assert all(line["output"].startswith(f"epoch {epoch}: subject ") for line in lines)
    # Found again, it names the same files
    [again] = run_experiment(experiment, _handles(writing, messages, tmp_path))
    assert again.reused == 5 and again.records[-1].record == evaluated


def test_the_report_sets_a_runs_checkpoints_one_under_the_other(
    writing, messages, tmp_path, write_experiment
):
    experiment = load(write_experiment(writing, "writing"))
    run_experiment(experiment, _handles(writing, messages, tmp_path))

    table = rows(scored(experiment, ledger_for(experiment)))
    assert [r.checkpoint for r in table] == ["epoch-1", "epoch-2"]
    header, body = headline(table)
    assert header[:4] == ["set", "approach", "checkpoint", "n"] and "ended" in header
    assert [cells[2] for cells in body] == ["epoch-1", "epoch-2"]
    # Columns nothing reported are left out
    assert "F1" not in header


def test_a_file_can_score_only_the_checkpoints_it_names(
    writing, messages, tmp_path, write_experiment
):
    experiment = load(write_experiment(writing, "writing", evaluate='checkpoints = ["epoch-1"]'))
    [result] = run_experiment(experiment, _handles(writing, messages, tmp_path))
    assert list(result.by_checkpoint) == ["epoch-1"]
    assert result.records[-1].request["checkpoints"] == ["epoch-1"]


def test_a_file_can_give_the_generation_settings_and_seed(
    writing, messages, tmp_path, write_experiment
):
    experiment = load(
        write_experiment(writing, "writing", evaluate="settings = { temperature = 0.5 }\nseed = 3")
    )
    [result] = run_experiment(experiment, _handles(writing, messages, tmp_path))
    request = result.records[-1].request
    assert (request["settings"], request["seed"]) == ({"temperature": 0.5}, 3)


def test_a_label_task_for_a_generator_is_refused_before_anything_trains(
    writing, messages, tmp_path, write_experiment
):
    experiment = load(
        write_experiment(writing, "writing", evaluate='tasks = [{ use = "classify" }]')
    )
    with pytest.raises(StageError, match="scores a label model's predictions"):
        run_experiment(experiment, _handles(writing, messages, tmp_path))
    assert not list((writing.root / "experiments").rglob("*.json"))


def test_a_generation_task_for_a_label_model_is_refused_before_anything_trains(
    project, catalog, experiment_file, tmp_path
):
    text = experiment_file.read_text().replace(
        'on = "holdout"\n', 'on = "holdout"\ntasks = [{ use = "text-generation" }]\n'
    )
    experiment_file.write_text(text)
    with pytest.raises(StageError, match="scores what a media model generates"):
        run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))


# ----------------------------------------------------------------------
# A label model, and keys
# ----------------------------------------------------------------------


def test_a_label_models_record_names_its_one_checkpoint(
    project, catalog, experiment_file, tmp_path
):
    [first, _] = run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))
    evaluated = first.records[-1].record
    assert [c["name"] for c in evaluated["checkpoints"]] == ["final"]
    assert evaluated["checkpoint"] == "final"
    assert first.by_checkpoint == {"final": first.metrics}


def test_a_file_using_none_of_the_later_fields_asks_as_it_did(
    project, catalog, experiment_file, tmp_path
):
    [first, _] = run_experiment(load(experiment_file), _handles(project, catalog, tmp_path))
    assert set(first.records[3].request) == {
        "dataset_dir",
        "dataset",
        "model",
        "params",
        "fresh_params",
        "fresh",
        "parent",
        "features",
        "experiment_id",
        "train_fraction",
        "subset_seed",
    }
    assert set(first.records[4].request) == {"run_id", "dataset_dir", "side", "tasks"}


def test_a_stage_key_from_before_checkpoints_is_unchanged():
    """Pinned: these keys are what the same stages were keyed as before. docs/adr/0037"""
    root = Path("/srv/projects/demo")
    experiment = Experiment(
        project="demo",
        project_id="demo",
        name="pinned",
        stages=[
            StageSpec(use="train", args={"fresh": True}),
            StageSpec(use="evaluate", args={"on": "holdout"}),
        ],
    )
    trained = TrainStageRequest(
        dataset_dir=root / "datasets/d/v1", model=f"{root / 'toy.py'}:Toy", fresh=True
    )
    evaluated = EvaluateRequest(
        run_id="20260101T000000000000-here", dataset_dir=root / "datasets/d/v1"
    )
    assert stage_key(
        experiment, 0, "1", portable(trained.model_dump(mode="json"), root), "cat-1"
    ) == ("66a28ce8652591df1eafc8c93ffb626d989aaf42e646b15437a0f0b4202df47b")
    assert stage_key(
        experiment, 1, "3", portable(evaluated.model_dump(mode="json"), root), "cat-1"
    ) == ("b828e38550c0af8046f68e451023a1fef0ff480bf49d169decbb9af3ff0639d0")


def test_a_warm_start_from_a_named_checkpoint_is_a_train_argument(
    project, catalog, experiment_file, tmp_path
):
    handles = _handles(project, catalog, tmp_path)
    [first, _] = run_experiment(load(experiment_file), handles)
    text = experiment_file.read_text().replace(
        "fresh = true\n", f'parent = "{first.run_id}"\nparent_checkpoint = "final"\n'
    )
    experiment_file.write_text(text)
    [warm, _] = run_experiment(load(experiment_file), handles)
    assert warm.records[3].request["parent_checkpoint"] == "final"
    assert warm.records[3].record["parent_run_id"] == first.run_id


def test_the_run_command_prints_each_checkpoints_scores(
    writing, messages, tmp_path, write_experiment, capsys
):
    config = tmp_path / "config.toml"
    config.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    assert main(["run", str(write_experiment(writing, "writing")), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "holdout text-generation @epoch-1:" in out
    assert "holdout text-generation @epoch-2:" in out
