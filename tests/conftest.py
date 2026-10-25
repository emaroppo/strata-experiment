"""A project over a catalog, with a model that needs no framework."""

import re
from pathlib import Path

import pytest

from strata.catalog import Catalog
from strata.contracts import Choices
from strata.project import Project

TOY = '''
import json
from pathlib import Path

from strata.contracts import ChoicesPrediction
from strata.modelling import LabelModel


class Toy(LabelModel):
    """Predicts "cat" for everything, so a score is a share of cats."""

    label_type = "classification"
    version = "1"

    def __init__(self, lr: float = 0.1, epochs: int = 1):
        self.lr, self.epochs = lr, epochs
        self.classes = []

    def finetune(self, train, classes, val=None, on_epoch=None):
        self.classes = list(classes)
        return {"n_train": float(len(train)), "n_val": float(len(val or [])), "lr": self.lr}

    def predict(self, paths, on_batch=None, *, features=None):
        return [ChoicesPrediction(values=["cat"], confidences=[0.9]) for _ in paths]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes}))

    def load(self, path):
        self.classes = json.loads(Path(path).read_text())["classes"]
'''


@pytest.fixture
def project(tmp_path, monkeypatch) -> Project:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "projects" / "demo"
    root.mkdir(parents=True)
    Project.create(root, name="demo", classes=["cat", "dog"])
    (root / "toy.py").write_text(TOY)
    toml = root / "project.toml"
    text = re.sub(r'^ref = ".*"$', 'ref = "toy.py:Toy"', toml.read_text(), flags=re.M)
    # The template's parameters are the image baseline's; the toy takes its own
    text = re.sub(r"^\[model\.params\]\n(?:\w+ = .*\n)*", "[model.params]\n", text, flags=re.M)
    text = re.sub(r"^\[model\.fresh_params\]\n(?:\w+ = .*\n)*", "", text, flags=re.M)
    toml.write_text(text)
    return Project.load(root)


@pytest.fixture
def catalog(project, tmp_path) -> Catalog:
    """Thirty samples, labelled cat or dog two to one, in ten groups of three."""
    catalog = Catalog.local(tmp_path / "catalog")
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = []
    for i in range(30):
        path = raw / f"img{i:02d}.jpg"
        path.write_bytes(f"image {i}".encode())
        paths.append(path)
    ids = []
    for group in range(10):
        ids += catalog.ingest(
            paths[group * 3 : group * 3 + 3],
            media="image",
            metadata={"video": f"g{group}"},
            collections=project.collections,
        )
    label_set = catalog.label_sets.create(project.label_set_name, project.label_set.schema)
    catalog.annotations.annotate_many(
        label_set, [(i, Choices(values=["cat" if n % 3 else "dog"])) for n, i in enumerate(ids)]
    )
    return catalog


@pytest.fixture
def write_experiment(tmp_path):
    """Writes an experiment file over a project: every stage once, ``train`` fresh, with what a
    test adds under ``train`` and ``evaluate`` and as its ``[grid]``.
    """

    def _write(
        project: Project,
        name: str,
        *,
        train: str = "",
        evaluate: str = "",
        grid: str = "",
        filename: str = "experiment.toml",
    ) -> Path:
        def block(lines: str) -> str:
            return f"{lines.rstrip()}\n" if lines.strip() else ""

        path = tmp_path / filename
        path.write_text(
            f'''
project = "{project.root}"
name = "{name}"

[[stage]]
use = "dataset"
val_ratio = 0.2
holdout_ratio = 0.2

[[stage]]
use = "materialise"

[[stage]]
use = "split"

[[stage]]
use = "train"
fresh = true
{block(train)}
[[stage]]
use = "evaluate"
on = "holdout"
{block(evaluate)}'''
            + (f"\n[grid]\n{block(grid)}" if grid.strip() else "")
        )
        return path

    return _write


@pytest.fixture
def experiment_file(project, write_experiment) -> Path:
    """A learning-rate grid of two over the toy model."""
    return write_experiment(
        project, "lr", train="params = { lr = 0.1 }", grid='"train.params.lr" = [0.1, 0.5]'
    )


@pytest.fixture
def config_file(catalog, tmp_path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f'[catalog]\nroot = "{tmp_path / "catalog"}"\n')
    return path
