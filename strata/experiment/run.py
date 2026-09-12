"""Running an experiment: deciding the calls, and recording what came back.

For each trial, for each stage: the key is computed from the spec alone;
if the ledger holds a record under it the record is reused, otherwise the
request is built — the file's arguments, the project's declarations, and
what upstream stages produced — the stage runs, and the record is written.
Nothing here trains or freezes anything itself.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from strata.catalog import Catalog
from strata.catalog import stages as catalog_stages
from strata.labeller.project import Project
from strata.modelling import RunStore
from strata.modelling import stages as modelling_stages
from strata.modelling.registry import absolute
from strata.modelling.stages import Host

from .ledger import Ledger, StageRecord, now, stage_key
from .registry import check, resolve
from .spec import Experiment, StageSpec, Trial


@dataclass
class Handles:
    """What the stages are handed: the project, its catalog, and the other host."""

    project: Project
    catalog: Catalog
    #: The catalog's local root, whose ``blobs/`` is the cache a materialise
    #: consults before the backend.
    catalog_root: Path
    host: Host | None = None
    on_progress: Any = None
    on_state: Any = None


@dataclass
class TrialResult:
    trial: Trial
    records: list[StageRecord] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        """What the last ``evaluate`` reported, or nothing."""
        for record in reversed(self.records):
            if record.stage == "evaluate":
                return dict(record.record.get("metrics", {}))
        return {}

    @property
    def run_id(self) -> str | None:
        for record in reversed(self.records):
            if record.stage == "train":
                return record.record.get("run_id")
        return None

    @property
    def reused(self) -> int:
        return sum(1 for r in self.records if r.reused)


#: Stage name -> the model its record parses back into, so a reused record
#: is handed downstream exactly as a fresh one would be.
RECORDS: dict[str, type[BaseModel]] = {
    "dataset": catalog_stages.DatasetRecord,
    "materialise": catalog_stages.MaterialiseRecord,
    "split": catalog_stages.SplitRecord,
    "train": modelling_stages.TrainRecord,
    "evaluate": modelling_stages.EvaluateRecord,
}


def run_experiment(
    experiment: Experiment,
    handles: Handles,
    ledger: Ledger | None = None,
    on_event=None,
) -> list[TrialResult]:
    """Every trial in order, each stage run or reused, everything recorded."""
    check(experiment)
    ledger = ledger or Ledger.under(handles.project.root)
    ledger.experiment_dir(experiment)
    say = on_event or (lambda *_: None)
    results = []
    for trial in experiment.trials():
        say("trial", trial, None)
        results.append(_run_trial(experiment, trial, handles, ledger, say))
    return results


def _run_trial(
    experiment: Experiment, trial: Trial, handles: Handles, ledger: Ledger, say
) -> TrialResult:
    result = TrialResult(trial=trial)
    produced: dict[str, BaseModel] = {}
    for index, spec in enumerate(trial.experiment.stages):
        stage, _ = resolve(spec.use)
        key = stage_key(trial.experiment, index, stage.version)
        found = ledger.find(key)
        if found is not None:
            record = RECORDS[spec.use].model_validate(found.record)
            found = found.model_copy(update={"reused": True})
            ledger.place(experiment, trial, found)
            produced[stage.produces] = record
            result.records.append(found)
            say("reused", trial, found)
            continue

        request = _request(spec, handles, produced)
        started = now()
        record = stage.run(request, _context(spec.use, handles))
        written = StageRecord(
            index=index,
            stage=stage.name,
            version=stage.version,
            key=key,
            request=request.model_dump(mode="json"),
            record=record.model_dump(mode="json"),
            started=started,
            finished=now(),
        )
        ledger.write(experiment, trial, written)
        produced[stage.produces] = record
        result.records.append(written)
        say("ran", trial, written)
    return result


def _context(name: str, handles: Handles):
    if name in ("dataset", "materialise", "split"):
        return catalog_stages.Context(
            handles.catalog,
            handles.project.datasets_dir,
            cache=handles.catalog_root / "blobs",
            on_progress=handles.on_progress,
        )
    return modelling_stages.Context(
        store=RunStore.local(handles.project.runs_dir),
        host=handles.host,
        on_state=handles.on_state,
    )


def _request(spec: StageSpec, handles: Handles, produced: dict[str, BaseModel]) -> BaseModel:
    """The stage's request: the file's arguments over the project's declarations,
    with what upstream produced wired in."""
    project = handles.project
    args = dict(spec.args)
    if spec.use == "dataset":
        return catalog_stages.DatasetRequest(
            label_set=project.label_set_name,
            collections=project.collections,
            **{"name": project.dataset_name, **args},
        )
    if spec.use == "materialise":
        return catalog_stages.MaterialiseRequest(
            dataset_id=produced["dataset_version"].dataset_id,
            features=[s.as_dict() for s in project.feature_specs],
        )
    if spec.use == "split":
        return catalog_stages.SplitRequest(dataset_dir=produced["dataset_dir"].directory, **args)
    if spec.use == "train":
        frozen = produced.get("dataset_version")
        identity = (
            modelling_stages.DatasetIdentity(
                dataset_id=frozen.dataset_id,
                name=frozen.name,
                version=frozen.version,
                annotation_digest=frozen.annotation_digest,
                catalog_id=frozen.catalog_id,
            )
            if frozen is not None
            else None
        )
        return modelling_stages.TrainStageRequest(
            dataset_dir=produced["dataset_dir"].directory,
            dataset=identity,
            # Anchored at the project, because a model.py belongs to the job
            model=absolute(args.pop("model", project.model.ref), project.root),
            # The file's parameters are the whole of them: a trial that says
            # `params` gets exactly those, not those plus the project's
            # cold-start extras. Absent, the project's own apply as a round's do.
            params=args.pop("params", project.model.params),
            fresh_params=args.pop(
                "fresh_params", {} if "params" in spec.args else project.model.fresh_params
            ),
            features=[s.as_dict() for s in project.feature_specs],
            **args,
        )
    if spec.use == "evaluate":
        return modelling_stages.EvaluateRequest(
            run_id=produced["run"].run_id,
            dataset_dir=produced["dataset_dir"].directory,
            side=args.get("on", "holdout"),
        )
    raise AssertionError(f"no wiring for {spec.use}: the registry and this table disagree")
