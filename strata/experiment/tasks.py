"""What an experiment file says to score with, as the evaluate stage takes it.

In the file, each task is a table naming it with ``use``, by name or a
project's own ``file.py:Class``, and every other key is its parameters::

    [[stage]]
    use = "evaluate"
    tasks = [
        { use = "entities" },
        { use = "mask", failures = ["fragmented", "failures.py:Initials"] },
    ]

A file reference, the task's or one inside its parameters, is anchored at
the project, as a model's is, and the key is made relative to it again, so
a project moved keeps its keys. Each task is resolved and checked against
the project's label set when the file is checked, and the identities found
go into the request, so a change in the code is a change in the key. See
``docs/adr/0043``.
"""

from typing import Any

from strata.modelling.plugins.registry import absolute
from strata.modelling.stages import TaskRef, resolved
from strata.project import Project

from .spec import Experiment, ExperimentError


def task_refs(entries: Any, project: Project) -> list[TaskRef] | None:
    """The file's tasks as requests, resolved against ``project``. None is the default."""
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
        params = {k: _anchored(v, project) for k, v in entry.items() if k != "use"}
        ref = TaskRef(ref=absolute(entry["use"], project.root), params=params)
        _, _, identities = resolved(ref, project.label_set.schema)
        refs.append(ref.model_copy(update={"identities": identities}))
    return refs


def check_tasks(experiment: Experiment, project: Project) -> None:
    """Refuse, before any stage runs, a trial whose tasks do not resolve or check."""
    for trial in experiment.trials():
        for spec in trial.experiment.stages:
            if spec.use == "evaluate":
                task_refs(spec.args.get("tasks"), project)


def _anchored(value: Any, project: Project) -> Any:
    """``value`` with every ``file.py:Class`` in it anchored at the project."""
    if isinstance(value, dict):
        return {k: _anchored(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [_anchored(v, project) for v in value]
    if isinstance(value, str) and ".py:" in value:
        return absolute(value, project.root)
    return value
