"""``strata-experiment``: run an experiment file, or check one.

Standard library only, like the other package commands. ``run`` resolves
the project and its catalog the way the labeller does, hands them to the
runner, and prints what each trial reported; ``check`` validates the file,
expands the grid and prints the trials without running anything.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata-experiment", description="An experiment, written down once and run."
    )
    parser.add_argument("--json", action="store_true", help="Print the records as JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Validate the file and list its trials")
    check.add_argument("file", type=Path)
    check.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    check.set_defaults(run=_check)

    run = commands.add_parser("run", help="Run every trial, skipping what the ledger holds")
    run.add_argument("file", type=Path)
    run.add_argument(
        "--config", type=Path, default=None, help="Host settings ($STRATA_CONFIG, else config.toml)"
    )
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.set_defaults(run=_run)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except Exception as e:
        if _is_refusal(e):
            print(str(e), file=sys.stderr)
            return 1
        raise


def _is_refusal(error: Exception) -> bool:
    from strata.catalog import CatalogError
    from strata.catalog.config import CatalogConfigError
    from strata.modelling.remote.client import RemoteError
    from strata.modelling.stages import StageError
    from strata.project import ProjectError

    from .registry import ChainError
    from .spec import ExperimentError

    return isinstance(
        error,
        (
            ExperimentError,
            ChainError,
            CatalogError,
            CatalogConfigError,
            ProjectError,
            StageError,
            RemoteError,
            ValueError,
        ),
    )


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise ValueError(f"--set takes KEY=VALUE, got {pair!r}")
        try:
            out[key] = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            out[key] = raw
    return out


def _check(args) -> int:
    from .registry import check
    from .spec import load

    experiment = load(args.file, _overrides(args.set))
    check(experiment)
    trials = experiment.trials()
    if args.json:
        print(
            json.dumps(
                {
                    "id": experiment.id,
                    "name": experiment.name,
                    "stages": [s.use for s in experiment.stages],
                    "trials": [{"id": t.id, "overrides": t.overrides} for t in trials],
                },
                indent=2,
                sort_keys=True,
            )
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
    from strata.catalog.config import open_catalog
    from strata.modelling.stages import Host
    from strata.project import Project, Settings

    from .run import Handles, run_experiment
    from .spec import load

    experiment = load(args.file, _overrides(args.set))
    project = Project.load(Path(experiment.project))
    settings = Settings.load(args.config)
    config = settings.catalogs.named(project.catalog.name)
    catalog = open_catalog(config)
    host = None
    if settings.modelling.url:
        if not settings.modelling.token:
            raise ValueError(
                "No token for the modelling host. Set $STRATA_MODELLING_TOKEN to the "
                "same value it was started with."
            )
        host = Host(settings.modelling.url, settings.modelling.token)
    handles = Handles(project=project, catalog=catalog, catalog_root=Path(config.root), host=host)

    def say(event, trial, record) -> None:
        if args.json:
            return
        if event == "trial":
            print(f"trial {trial.short_id}  {trial.label}")
        else:
            summary = _summary(record)
            print(f"  {record.index + 1:02d} {record.stage:<12} {event:<7}{summary}")

    results = run_experiment(experiment, handles, on_event=say)

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
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    print()
    for r in results:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(r.metrics.items()))
        print(f"{r.trial.short_id}  {r.trial.label:<40} {r.run_id or '-':<32} {metrics}")
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
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(body["metrics"].items()))
        return f"  {body['side']}: {metrics}"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
