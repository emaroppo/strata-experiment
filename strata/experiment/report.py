"""What experiments found, side by side, and on what.

Reads trials from the ledger without running anything. Trials that differ
only in a seed are one row: the mean, the spread and how many, since one
run of a trained model hides how much the next would differ. Every row says
which set it was scored on — a validation side is a dev number, and says
so, so it is never read as a held-out one.
"""

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .ledger import Ledger, StageRecord
from .spec import Experiment

#: The headline numbers, as (column, task, metric).
HEADLINE = [
    ("F1", "entities", "f1"),
    ("partial F1", "entities", "partial_f1"),
    ("leaked", "mask", "leaked"),
    ("fragmented", "mask", "fragmented"),
    ("merging", "mask", "merging"),
    ("char recall", "mask", "char_recall"),
]


@dataclass(frozen=True)
class Scored:
    """One trial that has been evaluated."""

    experiment: str
    approach: str
    model: str
    seed: object
    dataset: str
    version: int
    side: str
    samples: int
    metrics: dict[str, float]
    #: class -> {"f1": strict, "partial_f1": partial}
    per_class: dict[str, dict[str, float]]

    @property
    def set_label(self) -> str:
        if self.side == "val":
            return f"dev: {self.dataset} v{self.version} val, {self.samples}"
        return f"{self.side}: {self.dataset} v{self.version}, {self.samples}"


@dataclass
class Row:
    """Trials that differ only in their seed, summarised."""

    set_label: str
    approach: str
    trials: list[Scored] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trials)

    def summary(self, values: list[float]) -> tuple[float, float | None]:
        mean = statistics.fmean(values)
        return mean, (statistics.stdev(values) if len(values) > 1 else None)

    def metric(self, name: str) -> tuple[float, float | None] | None:
        values = [t.metrics[name] for t in self.trials if name in t.metrics]
        return self.summary(values) if values else None

    def per_class(self, label: str, kind: str) -> tuple[float, float | None] | None:
        values = [t.per_class[label][kind] for t in self.trials if label in t.per_class]
        return self.summary(values) if values else None

    @property
    def classes(self) -> list[str]:
        return sorted({c for t in self.trials for c in t.per_class})


def scored(experiment: Experiment, ledger: Ledger) -> list[Scored]:
    """Every trial of ``experiment`` the ledger holds an evaluation for."""
    found = []
    for trial in experiment.trials():
        directory = ledger.root / experiment.short_id / trial.short_id
        if not directory.is_dir():
            continue
        records = {r.stage: r for r in ledger.records(experiment, trial)}
        if "evaluate" not in records or "dataset" not in records:
            continue
        found.append(_scored(experiment, trial.overrides, records))
    return found


def _scored(experiment: Experiment, overrides: dict, records: dict[str, StageRecord]) -> Scored:
    dataset = records["dataset"].record
    evaluated = records["evaluate"].record
    train = records["train"].request if "train" in records else {}
    params = train.get("params") or {}
    distinguishing = {k: v for k, v in overrides.items() if not k.endswith(".seed")}
    approach = experiment.name + "".join(
        f"  {k.rsplit('.', 1)[-1]}={v}" for k, v in sorted(distinguishing.items())
    )
    metrics: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}
    for task, score in (evaluated.get("scores") or {}).items():
        for name, value in (score.get("metrics") or {}).items():
            metrics[f"{task}.{name}"] = value
        if task == "entities":
            for label, values in (score.get("per_class") or {}).items():
                per_class[label] = {
                    "f1": values.get("f1"),
                    "partial_f1": values.get("partial_f1"),
                }
    return Scored(
        experiment=experiment.name,
        approach=approach,
        model=str(train.get("model", "")),
        seed=params.get("seed"),
        dataset=dataset["name"],
        version=dataset["version"],
        side=evaluated["side"],
        samples=evaluated["samples"],
        metrics=metrics,
        per_class=per_class,
    )


def rows(trials: list[Scored]) -> list[Row]:
    """Trials grouped by set and approach, in the order first seen."""
    grouped: dict[tuple[str, str], Row] = {}
    for trial in trials:
        key = (trial.set_label, trial.approach)
        grouped.setdefault(key, Row(trial.set_label, trial.approach)).trials.append(trial)
    return list(grouped.values())


def _cell(value: tuple[float, float | None] | None) -> str:
    if value is None:
        return "-"
    mean, spread = value
    return f"{mean:.3f}" if spread is None else f"{mean:.3f} ± {spread:.3f}"


def headline(table: list[Row]) -> tuple[list[str], list[list[str]]]:
    header = ["set", "approach", "n", *(name for name, _, _ in HEADLINE)]
    body = [
        [
            row.set_label,
            row.approach,
            str(row.n),
            *(_cell(row.metric(f"{task}.{metric}")) for _, task, metric in HEADLINE),
        ]
        for row in table
    ]
    return header, body


def by_class(table: list[Row]) -> tuple[list[str], list[list[str]]]:
    """Strict and partial F1 per class, one cell each as "strict / partial"."""
    classes = sorted({c for row in table for c in row.classes})
    header = ["set", "approach", *classes]
    body = []
    for row in table:
        cells = []
        for label in classes:
            strict, partial = row.per_class(label, "f1"), row.per_class(label, "partial_f1")
            cells.append(
                "-" if strict is None else f"{strict[0]:.2f} / {partial[0] if partial else 0:.2f}"
            )
        body.append([row.set_label, row.approach, *cells])
    return header, body


def markdown(header: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def plain(header: list[str], body: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()
        for cells in [header, *body]
    )


def as_json(table: list[Row]) -> list[dict]:
    return [
        {
            "set": row.set_label,
            "approach": row.approach,
            "n": row.n,
            "seeds": [t.seed for t in row.trials],
            "metrics": _summaries(
                {
                    name: row.metric(name)
                    for name in sorted({m for t in row.trials for m in t.metrics})
                }
            ),
            "per_class": {
                label: _summaries(
                    {kind: row.per_class(label, kind) for kind in ("f1", "partial_f1")}
                )
                for label in row.classes
            },
        }
        for row in table
    ]


def _summaries(found: dict[str, tuple[float, float | None] | None]) -> dict[str, dict]:
    """Each (mean, spread) as a mapping, leaving out what no trial reported."""
    return {
        name: {"mean": value[0], "sd": value[1]}
        for name, value in found.items()
        if value is not None
    }


def ledger_for(experiment: Experiment) -> Ledger:
    return Ledger.under(Path(experiment.project))
