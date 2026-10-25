"""``strata-experiment``: run an experiment file, check one, or report on several. See
``docs/adr/0030``.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from strata.catalog import CatalogError
from strata.catalog.config import CatalogConfigError, open_catalog
from strata.common.cli import print_json, run
from strata.common.errors import Refused
from strata.modelling.remote.client import RemoteError
from strata.modelling.stages import Host, StageError
from strata.project import Project, ProjectError, Settings

from . import report
from .registry import ChainError, check
from .run import Handles, run_experiment
from .spec import ExperimentError, load
from .tasks import check_tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata-experiment", description="An experiment, written down once and run."
    )
    parser.add_argument("--json", action="store_true", help="Print the records as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    checker = commands.add_parser("check", help="Validate the file and list its trials")
    checker.add_argument("file", type=Path)
    checker.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    checker.set_defaults(run=_check)

    runner = commands.add_parser("run", help="Run every trial, skipping what the ledger holds")
    runner.add_argument("file", type=Path)
    runner.add_argument(
        "--config", type=Path, default=None, help="Host settings ($STRATA_CONFIG, else config.toml)"
    )
    runner.add_argument(
        "--host",
        default="",
        help=(
            "Which modelling host in config.toml "
            "(default: the project's, else its catalog's, else the machine's)"
        ),
    )
    runner.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    runner.set_defaults(run=_run)

    reporter = commands.add_parser(
        "report", help="Set what experiment files found side by side, from the ledger"
    )
    reporter.add_argument("files", type=Path, nargs="+")
    reporter.add_argument("--markdown", action="store_true", help="Print Markdown tables")
    reporter.set_defaults(run=_report)

    args = parser.parse_args(argv)
    return run(lambda: args.run(args), *_refusals())


def _refusals() -> tuple[type[Exception], ...]:
    """What the command stops for with a line on stderr; anything else is a crash."""
    return (
        ExperimentError,
        ChainError,
        CatalogError,
        CatalogConfigError,
        ProjectError,
        StageError,
        RemoteError,
    )


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise Refused(f"--set takes KEY=VALUE, got {pair!r}")
        try:
            out[key] = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            out[key] = raw
    return out


def _ticker(label: str):
    """A one-line progress display on stderr, for a terminal only.

    Returns ``(show, done)``: ``show(text)`` redraws the line, ``done()`` ends it, so what is
    printed next starts a line of its own. Under ``--json`` stdout stays the record, since this
    never writes there. See ``docs/adr/0031``.
    """
    if not sys.stderr.isatty():
        return (lambda text: None), (lambda: None)
    shown = False

    def show(text: str) -> None:
        nonlocal shown
        shown = True
        print(f"\r{label} {text}", end="", file=sys.stderr, flush=True)

    def done() -> None:
        nonlocal shown
        if shown:
            shown = False
            print(file=sys.stderr, flush=True)

    return show, done


def _check(args) -> int:
    experiment = load(args.file, _overrides(args.set))
    check(experiment)
    check_tasks(experiment, Project.load(Path(experiment.project)))
    trials = experiment.trials()
    if args.json:
        print_json(
            {
                "id": experiment.id,
                "name": experiment.name,
                "stages": [s.use for s in experiment.stages],
                "trials": [{"id": t.id, "overrides": t.overrides} for t in trials],
            }
        )
        return 0
    print(f"{experiment.name}  {experiment.short_id}")
    print(f"  project  {experiment.project}")
    print(f"  stages   {' -> '.join(s.use for s in experiment.stages)}")
    print(f"  trials   {len(trials)}")
    for trial in trials:
        print(f"    {trial.short_id}  {trial.label}")
    return 0


def _run(args) -> int:
    experiment = load(args.file, _overrides(args.set))
    project = Project.load(Path(experiment.project))
    settings = Settings.load(args.config)
    config = settings.catalogs.named(project.catalog.name)
    catalog = open_catalog(config)
    host = None
    modelling = settings.modelling_for(project.catalog.name, args.host or project.model.host)
    if modelling.url:
        if not modelling.token:
            raise Refused(
                f"No token for the modelling host {modelling.describe()}. Set "
                f"{modelling.token_from()} to the same value it was started with."
            )
        host = Host(modelling.url, modelling.token)

    show, done = _ticker("running")

    def state(job: dict) -> None:
        if job.get("state") == "unreachable":
            show(f"host unreachable (attempt {job['attempts']}), the round is unaffected")
            return
        stage = job.get("stage", job.get("state", ""))
        if job.get("total"):
            stage = f"{stage} {job['done']:,}/{job['total']:,}"
        show(f"host: {stage}")

    handles = Handles(
        project=project,
        catalog=catalog,
        catalog_root=Path(config.root),
        host=host,
        on_progress=lambda fetched, total: show(f"materialising {fetched}/{total}"),
        on_state=state,
        on_epoch=lambda epoch, total, _metrics: show(f"epoch {epoch}/{total}"),
    )

    def say(event, trial, record) -> None:
        done()
        if args.json:
            return
        if event == "trial":
            print(f"trial {trial.short_id}  {trial.label}")
        else:
            summary = _summary(record)
            print(f"  {record.index + 1:02d} {record.stage:<12} {event:<7}{summary}")

    try:
        results = run_experiment(experiment, handles, on_event=say)
    finally:
        done()

    if args.json:
        payload = [
            {
                "trial": r.trial.id,
                "overrides": r.trial.overrides,
                "run_id": r.run_id,
                "metrics": r.metrics,
                "reused": r.reused,
                "records": [rec.model_dump(mode="json") for rec in r.records],
            }
            for r in results
        ]
        print_json(payload)
        return 0

    print()
    for r in results:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(r.metrics.items()))
        print(f"{r.trial.short_id}  {r.trial.label:<40} {r.run_id or '-':<32} {metrics}")
    return 0


def _report(args) -> int:
    trials = []
    for path in args.files:
        experiment = load(path)
        trials += report.scored(experiment, report.ledger_for(experiment))
    table = report.rows(trials)
    if args.json:
        print_json(report.as_json(table))
        return 0
    if not table:
        print("Nothing evaluated yet in these files: run them first.")
        return 0
    show = report.markdown if args.markdown else report.plain
    print(show(*report.headline(table)))
    print()
    print(
        "Per class, strict / partial F1:"
        if not args.markdown
        else "Per class, strict / partial F1:\n"
    )
    print(show(*report.by_class(table)))
    return 0


def _summary(record) -> str:
    body = record.record
    if record.stage == "dataset":
        note = f"  {body['name']} v{body['version']}, {body['samples']} sample(s)"
        if body.get("given"):
            note += f", {body['given']} side(s) given"
        if body.get("groups_cut"):
            note += f", {body['groups_cut']} group(s) cut by them"
        return note
    if record.stage == "materialise":
        return (
            f"  {body['train']} train, {body['val']} val, {body['holdout']} holdout"
            + (f", {body['fetched']} fetched" if body["fetched"] else "")
            + (
                f", {body['disputed']} left out over a disputed feature"
                if body.get("disputed")
                else ""
            )
        )
    if record.stage == "split":
        return "  drawn" if body["drawn"] else "  inherited"
    if record.stage == "train":
        return f"  run {body['run_id']}" + (" (cold)" if not body["parent_run_id"] else "")
    if record.stage == "evaluate":
        # A line per checkpoint where there were several. docs/adr/0054
        checkpoints = body.get("checkpoints") or []
        scored = (
            [(c["name"], c["scores"]) for c in checkpoints]
            if len(checkpoints) > 1
            else [(None, body.get("scores") or {"": {"metrics": body.get("metrics", {})}})]
        )
        lines = []
        for checkpoint, scores in scored:
            for task, score in scores.items():
                metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(score["metrics"].items()))
                label = f"{body['side']} {task}" if task else body["side"]
                if checkpoint is not None:
                    label += f" @{checkpoint}"
                lines.append(f"  {label}: {metrics}")
        return "\n".join(lines)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
