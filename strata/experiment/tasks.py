"""What an experiment file says to score with, as the evaluate stage takes it.

Each task is a table with ``use`` and its parameters; file references are
anchored at the project, and identities go into the request. See ``docs/adr/0042`` and
``docs/adr/0054``.
"""

from typing import Any

from strata.common.plugins import anchor_all
from strata.modelling.plugins.registry import absolute
from strata.modelling.stages import TaskRef, makes, resolved
from strata.project import Project

from .spec import Experiment, ExperimentError


def task_refs(entries: Any, project: Project, media: str | None = None) -> list[TaskRef] | None:
    """The file's tasks as requests, resolved against ``project``. None is the default.

    ``media`` is what the model scored generates, or None for a label model.
    """
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ExperimentError(
            f"evaluate's tasks is a list of tables, each naming a task with `use`; got {entries!r}."
        )
    refs = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("use"), str):
            raise ExperimentError(
                f"evaluate's task {index} names no task: each is a table with `use`, e.g. "
                f'{{ use = "mask" }}; got {entry!r}.'
            )
        params = {k: anchor_all(v, project.root) for k, v in entry.items() if k != "use"}
        ref = TaskRef(ref=absolute(entry["use"], project.root), params=params)
        _, _, identities = resolved(ref, project.label_set.schema, media)
        refs.append(ref.model_copy(update={"identities": identities}))
    return refs


def check_tasks(experiment: Experiment, project: Project) -> None:
    """Refuse, before any stage runs, a trial whose tasks do not resolve or check."""
    for trial in experiment.trials():
        model = project.model_ref()
        for spec in trial.experiment.stages:
            if spec.use == "train":
                model = project.model_ref(spec.args.get("model"))
            if spec.use == "evaluate" and spec.args.get("tasks") is not None:
                task_refs(spec.args["tasks"], project, makes(model))
