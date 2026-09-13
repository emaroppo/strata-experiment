"""What ran, written down where a person reads it as often as a program does.

Under the project's ``experiments/``: one ``records/`` directory holding
every stage record by its key, and one directory per experiment with one
per trial, where each stage's record is copied under its position. The key
is the hash of the canonical spec through that stage, the version of the
implementation that ran it, the request as it was actually built, and the
catalog it was built against — so two trials, or two experiments, that
agree on all of that through a stage share its record, and a stage whose
key already has one is not run again. Nothing is ever invalidated: a
changed argument, a changed project, or a rerun upstream changes every key
after it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from strata.common.canonical import content_hash

from .spec import Experiment, Trial

EXPERIMENTS_DIR = "experiments"
RECORDS_DIR = "records"


class StageRecord(BaseModel):
    """One stage that ran, or was found already run."""

    model_config = ConfigDict(extra="forbid")

    index: int
    stage: str
    version: str
    #: The hash of the canonical spec through this stage, plus the version.
    key: str
    request: dict[str, Any]
    record: dict[str, Any]
    started: datetime
    finished: datetime
    #: True in a trial's copy when the trial found the record rather than
    #: made it. Never true in ``records/``, which holds the original.
    reused: bool = False


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    overrides: dict[str, Any]
    spec: dict[str, Any] = Field(default_factory=dict)


def stage_key(
    experiment: Experiment, index: int, version: str, request: dict, catalog_id: str | None
) -> str:
    """The identity of a stage's position: everything upstream, what ran it,
    what it was asked, and where.

    ``request`` is the effective request in portable form. It carries what
    the project supplied and what upstream produced, so an edit to
    ``project.toml`` or a rerun upstream moves the key rather than reusing
    a record made under the old inputs. ``catalog_id`` is the catalog the
    stages were handed, so a record made against one catalog is never
    handed to a run over another.
    """
    return content_hash(
        {
            "prefix": experiment.prefix(index),
            "version": version,
            "request": request,
            "catalog": catalog_id,
        }
    )


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.records_dir = self.root / RECORDS_DIR

    @classmethod
    def under(cls, project_root: Path) -> "Ledger":
        return cls(Path(project_root) / EXPERIMENTS_DIR)

    def experiment_dir(self, experiment: Experiment) -> Path:
        directory = self.root / experiment.short_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "experiment.json"
        if not path.exists():
            path.write_text(
                _dumps(
                    {
                        "id": experiment.id,
                        "name": experiment.name,
                        "project_as_written": experiment.project,
                        **experiment.canonical(),
                    }
                )
            )
        return directory

    def trial_dir(self, experiment: Experiment, trial: Trial) -> Path:
        directory = self.experiment_dir(experiment) / trial.short_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trial.json"
        if not path.exists():
            record = TrialRecord(
                id=trial.id, overrides=trial.overrides, spec=trial.experiment.canonical()
            )
            path.write_text(_dumps(record.model_dump(mode="json")))
        return directory

    def find(self, key: str) -> StageRecord | None:
        """The record written under this key, wherever it was made."""
        path = self.records_dir / f"{key}.json"
        if not path.exists():
            return None
        return StageRecord.model_validate_json(path.read_text())

    def write(self, experiment: Experiment, trial: Trial, record: StageRecord) -> Path:
        """Keep the original by key, and a copy where the trial can be read whole."""
        self.records_dir.mkdir(parents=True, exist_ok=True)
        original = record.model_copy(update={"reused": False})
        (self.records_dir / f"{record.key}.json").write_text(
            _dumps(original.model_dump(mode="json"))
        )
        return self.place(experiment, trial, record)

    def place(self, experiment: Experiment, trial: Trial, record: StageRecord) -> Path:
        """A trial's copy of a record, under its position, reused or not."""
        name = trial.experiment.stages[record.index].use
        path = self.trial_dir(experiment, trial) / f"{record.index + 1:02d}-{name}.json"
        path.write_text(_dumps(record.model_dump(mode="json")))
        return path

    def records(self, experiment: Experiment, trial: Trial) -> list[StageRecord]:
        directory = self.trial_dir(experiment, trial)
        return [
            StageRecord.model_validate_json(path.read_text())
            for path in sorted(directory.glob("[0-9][0-9]-*.json"))
        ]


def now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def _dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
