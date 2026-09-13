"""Names to stages, and what each is handed from where.

A hand-maintained table rather than an entry-point group: the orchestrator
imports catalog and modelling regardless, there is no third-party stage,
and a plugin seam is designed at the third implementation. Readable in one
place, which is what a config can name.

Each entry says which of a stage's request fields the file supplies and
which the orchestrator wires in from the project or from an upstream
record. The chain is checked before anything runs: every kind a stage
consumes must have been produced by a stage before it.
"""

from strata.catalog import stages as catalog_stages
from strata.common.stages import Stage
from strata.modelling import stages as modelling_stages

from .spec import Experiment


class ChainError(ValueError):
    """A sequence of stages that cannot hold, found before the first one runs."""


#: Stage name -> (declaration, the argument keys the file may set).
STAGES: dict[str, tuple[Stage, frozenset[str]]] = {
    "dataset": (
        catalog_stages.STAGES[0],
        frozenset({"name", "val_ratio", "holdout_ratio", "seed", "group_by"}),
    ),
    "materialise": (catalog_stages.STAGES[1], frozenset()),
    "split": (
        catalog_stages.STAGES[2],
        frozenset({"seed", "val_ratio", "holdout_ratio", "group_by"}),
    ),
    "train": (
        modelling_stages.STAGES[0],
        frozenset({"model", "params", "fresh_params", "fresh", "parent"}),
    ),
    "evaluate": (modelling_stages.STAGES[1], frozenset({"on"})),
}
assert [s.name for s, _ in STAGES.values()] == list(STAGES), "the table names its own stages"


def resolve(name: str) -> tuple[Stage, frozenset[str]]:
    try:
        return STAGES[name]
    except KeyError:
        raise ChainError(f"No stage named {name!r}. Registered: {', '.join(STAGES)}.") from None


def check(experiment: Experiment) -> None:
    """Refuse a file whose stages cannot follow each other or take what they are given.

    Walks the declared kinds the way a schema is walked through a
    processing pipeline: a stage consuming a kind nothing before it
    produced fails here, naming both, rather than an hour in.
    """
    have: set[str] = set()
    for index, spec in enumerate(experiment.stages):
        stage, allowed = resolve(spec.use)
        unknown = sorted(set(spec.args) - allowed)
        if unknown:
            raise ChainError(
                f"Stage {index} ({spec.use}) does not take {', '.join(unknown)}. "
                f"It takes: {', '.join(sorted(allowed)) or 'nothing'}."
            )
        missing = [kind for kind in stage.consumes if kind not in have]
        if missing:
            raise ChainError(
                f"Stage {index} ({spec.use}) needs a {missing[0]} and nothing before it "
                f"produces one. Produced so far: {', '.join(sorted(have)) or 'nothing'}."
            )
        have.add(stage.produces)

    if experiment.grid:
        train = [s for s in experiment.stages if s.use == "train"]
        for spec in train:
            if not spec.args.get("fresh") and not spec.args.get("parent"):
                raise ChainError(
                    "A grid needs every trial cold, or every trial continued from one named "
                    "parent: set `fresh = true` or `parent = \"<run id>\"` on the train "
                    "stage. Warm-starting each trial from the previous one would make the "
                    "results depend on the order they ran in."
                )
