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
``docs/adr/0042``.

A media model's run is scored by a task that reads what it generates, and
a label model's by one that reads its label set: each is checked against
the model the train stage before it names, so a task for the other kind is
refused before anything trains (``docs/adr/0054``).
"""

from typing import Any

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
        params = {k: _anchored(v, project) for k, v in entry.items() if k != "use"}
        ref = TaskRef(ref=absolute(entry["use"], project.root), params=params)
        _, _, identities = resolved(ref, project.label_set.schema, media)
        refs.append(ref.model_copy(update={"identities": identities}))
    return refs


def check_tasks(experiment: Experiment, project: Project) -> None:
    """Refuse, before any stage runs, a trial whose tasks do not resolve or check."""
    for trial in experiment.trials():
        # The model of the train stage before, as the request will name it
        model = project.model_ref()
        for spec in trial.experiment.stages:
            if spec.use == "train":
                model = project.model_ref(spec.args.get("model"))
            if spec.use == "evaluate" and spec.args.get("tasks") is not None:
                task_refs(spec.args["tasks"], project, makes(model))


def _anchored(value: Any, project: Project) -> Any:
    """``value`` with every ``file.py:Class`` in it anchored at the project."""
    if isinstance(value, dict):
        return {k: _anchored(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [_anchored(v, project) for v in value]
    if isinstance(value, str) and ".py:" in value:
        return absolute(value, project.root)
    return value
