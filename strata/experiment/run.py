"""Running an experiment: building each stage's request, reusing the ledger's record under its key
or running and recording.

See ``docs/adr/0037``.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from strata.catalog import Catalog, SelectionSpec
from strata.catalog import stages as catalog_stages
from strata.modelling import RunStore
from strata.modelling import stages as modelling_stages
from strata.modelling.stages import Host
from strata.project import Project

from .ledger import Ledger, StageRecord, now, stage_key
from .registry import check, resolve
from .spec import Experiment, ExperimentError, StageSpec, Trial
from .tasks import check_tasks, task_refs


@dataclass
class Handles:
    """What the stages are handed: the project, its catalog, and the other host."""

    project: Project
    catalog: Catalog
    #: The catalog's local root, whose ``blobs/`` is the materialise cache (docs/adr/0002).
    catalog_root: Path
    host: Host | None = None
    #: ``on_progress(done, total)`` as a catalog stage fetches blobs.
    on_progress: Callable[[int, int], None] | None = None
    #: A remote job's state as a modelling stage polls it.
    on_state: Callable[[dict], None] | None = None
    #: ``on_epoch(done, total, metrics)`` as a model trains here, the modelling stage's
    #: ``EpochReport``; it may answer True to ask the model to stop.
    on_epoch: Callable[[int, int, dict[str, float]], bool | None] | None = None


@dataclass
class TrialResult:
    trial: Trial
    records: list[StageRecord] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        """What the last ``evaluate`` reported, ``<task>.<metric>``, or nothing; a record from
        before tasks as it was.
        """
        for record in reversed(self.records):
            if record.stage == "evaluate":
                scores = record.record.get("scores")
                if not scores:
                    return dict(record.record.get("metrics", {}))
                return {
                    f"{task}.{name}": value
                    for task, score in scores.items()
                    for name, value in score.get("metrics", {}).items()
                }
        return {}

    @property
    def by_checkpoint(self) -> dict[str | None, dict[str, float]]:
        """What the last ``evaluate`` reported at each checkpoint, oldest first; one entry under
        None before checkpoints.

        See ``docs/adr/0054``.
        """
        for record in reversed(self.records):
            if record.stage == "evaluate":
                checkpoints = record.record.get("checkpoints")
                if not checkpoints:
                    return {None: self.metrics}
                return {
                    c["name"]: {
                        f"{task}.{name}": value
                        for task, score in c.get("scores", {}).items()
                        for name, value in score.get("metrics", {}).items()
                    }
                    for c in checkpoints
                }
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


#: Stage name -> the model its record parses back into. docs/adr/0037
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
    check_tasks(experiment, handles.project)
    if experiment.project_id != handles.project.name:
        raise ExperimentError(
            f"The file was loaded for project {experiment.project_id!r} and is being run "
            f"over {handles.project.name!r}."
        )
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
    catalog_id = handles.catalog.id
    for index, spec in enumerate(trial.experiment.stages):
        stage, _ = resolve(spec.use)
        request = _request(spec, handles, produced, experiment.id)
        asked = portable(request.model_dump(mode="json"), handles.project.root)
        if isinstance(request, catalog_stages.DatasetRequest) and request.spec is not None:
            # The key follows what the spec selects, not only where it is. docs/adr/0050
            asked["spec"] += f"#{SelectionSpec.read(request.spec).spec_hash}"
        key = stage_key(trial.experiment, index, stage.version, asked, catalog_id)
        found = ledger.find(key)
        if found is not None:
            record = RECORDS[spec.use].model_validate(found.record)
            found = found.model_copy(update={"reused": True})
            ledger.place(experiment, trial, found)
            produced[stage.produces] = record
            result.records.append(found)
            say("reused", trial, found)
            continue

        started = now()
        record = stage.run(request, _context(spec.use, handles, ledger))
        written = StageRecord(
            index=index,
            stage=stage.name,
            version=stage.version,
            key=key,
            request=asked,
            record=record.model_dump(mode="json"),
            started=started,
            finished=now(),
        )
        ledger.write(experiment, trial, written)
        produced[stage.produces] = record
        result.records.append(written)
        say("ran", trial, written)
    return result


def _context(name: str, handles: Handles, ledger: Ledger):
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
        on_epoch=handles.on_epoch,
        on_state=handles.on_state,
        # docs/adr/0054
        outputs=ledger.outputs_dir,
    )


def portable(payload: Any, root: Path) -> Any:
    """``payload`` with every path under ``root``, plain or in a ``file:`` identity, made relative
    to it.

    See ``docs/adr/0037`` and ``docs/adr/0042``.
    """
    prefix = str(Path(root).resolve()) + os.sep
    marked = f"file:{prefix}"

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix) :]
        if isinstance(value, str) and value.startswith(marked):
            return "file:" + value[len(marked) :]
        return value

    return walk(payload)


R = TypeVar("R", bound=BaseModel)


def _produced[R: BaseModel](produced: dict[str, BaseModel], kind: str, cls: type[R]) -> R:
    """What an upstream stage produced under ``kind``, as the record it must be. See
    ``docs/adr/0037``.
    """
    record = produced[kind]
    if not isinstance(record, cls):
        raise ExperimentError(
            f"{kind!r} was produced as {type(record).__name__}, not {cls.__name__}"
        )
    return record


def _materialised(produced: dict[str, BaseModel]):
    """The dataset directory, whichever stage last produced one."""
    record = produced["dataset_dir"]
    if not isinstance(record, (catalog_stages.MaterialiseRecord, catalog_stages.SplitRecord)):
        raise ExperimentError(f"'dataset_dir' was produced as {type(record).__name__}")
    return record.directory


def _frozen(produced: dict[str, BaseModel]):
    return _produced(produced, "dataset_version", catalog_stages.DatasetRecord)


def _request(
    spec: StageSpec, handles: Handles, produced: dict[str, BaseModel], experiment_id: str
) -> BaseModel:
    """The stage's request: the file's arguments over the project's declarations, with what upstream
    produced.
    """
    project = handles.project
    args = dict(spec.args)
    # The project's grouping unless the file says otherwise, including none. docs/adr/0024
    grouping = {"group_by": project.catalog.group_by or None}
    if spec.use == "dataset":
        declared = {**grouping, "given": project.catalog.given_split}
        if "spec" in args:
            # Anchored at the project, wherever the experiment runs from.
            args["spec"] = (project.root / args["spec"]).resolve()
        return catalog_stages.DatasetRequest(
            label_set=project.label_set_name,
            collections=project.collections,
            **{"name": project.dataset_name, **declared, **args},
        )
    if spec.use == "materialise":
        return catalog_stages.MaterialiseRequest(
            dataset_id=_frozen(produced).dataset_id,
            features=[s.as_dict() for s in project.feature_specs],
        )
    if spec.use == "split":
        return catalog_stages.SplitRequest(
            dataset_dir=_materialised(produced),
            **{**grouping, **args},
        )
    if spec.use == "train":
        frozen = _frozen(produced) if "dataset_version" in produced else None
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
        # The file's params over the project's; another model starts from nothing. docs/adr/0037
        model = args.pop("model", project.model.ref)
        own = model == project.model.ref
        params = {**(project.model.params if own else {}), **args.pop("params", {})}
        fresh_params = args.pop("fresh_params", project.model.fresh_params if own else {})
        return modelling_stages.TrainStageRequest(
            dataset_dir=_materialised(produced),
            dataset=identity,
            model=project.model_ref(model),
            params=params,
            fresh_params=fresh_params,
            features=[s.as_dict() for s in project.feature_specs],
            experiment_id=experiment_id,
            **args,
        )
    if spec.use == "evaluate":
        run = _produced(produced, "run", modelling_stages.TrainRecord)
        named = args.get("tasks")
        # docs/adr/0054
        media = modelling_stages.makes(run.model) if named is not None else None
        # Later fields only where the file gives them, so older keys hold. docs/adr/0037
        later = {k: args[k] for k in ("checkpoints", "settings", "seed") if k in args}
        return modelling_stages.EvaluateRequest(
            run_id=run.run_id,
            dataset_dir=_materialised(produced),
            side=args.get("on", "holdout"),
            tasks=task_refs(named, project, media),
            **later,
        )
    raise AssertionError(f"no wiring for {spec.use}: the registry and this table disagree")
