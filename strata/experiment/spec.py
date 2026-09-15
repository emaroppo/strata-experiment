"""The experiment file, and the hashes that make it an identity.

TOML is what a person edits; canonical JSON is what is hashed. Every
default is materialised before hashing, and a schema version sits inside
the payload so a change to what a field means changes the hash even when
the text does not. A trial is the base file plus overrides, applied before
validation and hashing. The file names its project by how to find it; the
hash covers the project's identity instead, bound at load. See
``docs/adr/0037``.
"""

import copy
import itertools
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata.common.canonical import content_hash, short_hash

SCHEMA_VERSION = 1


class ExperimentError(ValueError):
    """A file that cannot be run as written."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageSpec(Strict):
    """One stage as the file names it: a registered name and literal arguments."""

    use: str
    args: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_table(cls, table: dict) -> "StageSpec":
        if "use" not in table:
            raise ExperimentError(f"A [[stage]] table needs `use`, the stage's name. Got: {table}")
        rest = {k: v for k, v in table.items() if k != "use"}
        return cls(use=table["use"], args=rest)


class Experiment(Strict):
    """One named, reproducible sequence over a project, with its variation."""

    schema_version: int = SCHEMA_VERSION
    #: How to find the project, as the file wrote it: a name under
    #: ``projects/`` or a path. Not in the hash.
    project: str
    #: The project's identity, which the hash covers: its name, as
    #: ``project.toml`` declares it. Bound at load, never written in the
    #: file. docs/adr/0037
    project_id: str
    name: str
    stages: list[StageSpec]
    #: ``"<stage>.<dotted.key>"`` to the values it takes, one trial per
    #: combination. Empty means one trial: the file as written.
    grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _grid_names_stages_once(self):
        if not self.stages:
            raise ExperimentError("An experiment needs at least one stage.")
        names = [s.use for s in self.stages]
        for key, values in self.grid.items():
            stage, _, path = key.partition(".")
            if not path:
                raise ExperimentError(
                    f"A grid key names a stage and a key inside it, `train.params.lr`; got {key!r}."
                )
            if stage not in names:
                raise ExperimentError(f"Grid key {key!r} names a stage this file does not use.")
            if names.count(stage) > 1:
                raise ExperimentError(
                    f"Grid key {key!r} is ambiguous: `{stage}` appears {names.count(stage)} "
                    f"times. A grid can only vary a stage that appears once."
                )
            if not values:
                raise ExperimentError(f"Grid key {key!r} lists no values.")
        return self

    def canonical(self) -> dict[str, Any]:
        """The form that gets hashed: every default materialised, nothing implied.

        The project is in it by identity, not by the path the file used to
        find it.
        """
        payload = self.model_dump(mode="python")
        payload.pop("project")
        return payload

    @property
    def id(self) -> str:
        return content_hash(self.canonical())

    @property
    def short_id(self) -> str:
        return short_hash(self.canonical(), 12)

    def prefix(self, upto: int) -> dict[str, Any]:
        """What stage ``upto`` sees: the project, and every stage through it.

        The grid is not in it: a trial's stages already carry their
        overridden values. See ``docs/adr/0037``.
        """
        return {
            "schema_version": self.schema_version,
            "project": self.project_id,
            "stages": [s.model_dump(mode="python") for s in self.stages[: upto + 1]],
        }

    def prefix_hash(self, upto: int) -> str:
        return content_hash(self.prefix(upto))

    def trials(self) -> list["Trial"]:
        """One trial per combination of the grid, in the order listed."""
        if not self.grid:
            return [Trial(overrides={}, experiment=self)]
        keys = list(self.grid)
        trials = []
        for combination in itertools.product(*(self.grid[k] for k in keys)):
            overrides = dict(zip(keys, combination, strict=True))
            trials.append(Trial(overrides=overrides, experiment=self.with_overrides(overrides)))
        return trials

    def with_overrides(self, overrides: dict[str, Any], keep_grid: bool = False) -> "Experiment":
        """This file with each ``stage.dotted.key`` set.

        A trial drops the grid, since its values are now written in. An
        override given at load keeps it, since the grid is still to expand —
        minus any key the override pins, because a value said outright
        beats a list of candidates for it.
        """
        # Deep, or two trials would write into one nested dict and share it
        stages = [copy.deepcopy(s.args) for s in self.stages]
        names = [s.use for s in self.stages]
        for key, value in overrides.items():
            stage, _, path = key.partition(".")
            _assign(stages[names.index(stage)], path.split("."), value, key)
        return self.model_copy(
            update={
                "stages": [
                    StageSpec(use=name, args=args) for name, args in zip(names, stages, strict=True)
                ],
                "grid": (
                    {k: v for k, v in self.grid.items() if k not in overrides} if keep_grid else {}
                ),
            }
        )


class Trial(Strict):
    """One point of the grid: the overrides, and the file with them applied."""

    overrides: dict[str, Any]
    experiment: Experiment

    @property
    def id(self) -> str:
        return self.experiment.id

    @property
    def short_id(self) -> str:
        return self.experiment.short_id

    @property
    def label(self) -> str:
        if not self.overrides:
            return "as written"
        return ", ".join(f"{k}={v}" for k, v in self.overrides.items())


def load(path: Path, overrides: dict[str, Any] | None = None) -> Experiment:
    """Read an experiment file, applying ``overrides`` before validation.

    The project the file names is loaded here, for its identity. See
    ``docs/adr/0037``.
    """
    with open(path, "rb") as f:
        payload = tomllib.load(f)
    locator = payload.get("project")
    if not isinstance(locator, str) or not locator:
        raise ExperimentError("An experiment file needs `project`, the project it runs over.")
    return from_payload(payload, overrides, project_id=identity_of(locator))


def identity_of(locator: str) -> str:
    """The identity of the project ``locator`` finds: its declared name."""
    from strata.project import Project

    return Project.load(Path(locator)).name


def from_payload(
    payload: dict, overrides: dict[str, Any] | None = None, *, project_id: str
) -> Experiment:
    payload = dict(payload)
    if "project_id" in payload:
        raise ExperimentError(
            "`project_id` is not a key of the file: the identity comes from the project "
            "that `project` names."
        )
    stages = payload.pop("stage", None)
    if stages is None:
        raise ExperimentError("An experiment file needs at least one [[stage]] table.")
    try:
        experiment = Experiment(
            stages=[StageSpec.from_table(s) for s in stages], project_id=project_id, **payload
        )
    except (ValueError, TypeError) as e:
        raise ExperimentError(str(e)) from None
    if overrides:
        experiment = experiment.with_overrides(overrides, keep_grid=True)
    return experiment


def _assign(target: dict, path: list[str], value: Any, key: str) -> None:
    for part in path[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[path[-1]] = value
