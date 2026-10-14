"""Reporting on experiments that have run: grouped over seeds, and labelled by set."""

import json

from strata.experiment import load
from strata.experiment.cli import main
from strata.experiment.report import Scored, as_json, by_class, headline, ledger_for, rows, scored


def trial(approach, seed, f1, partial, side="val", per_class=None):
    return Scored(
        experiment="x",
        approach=approach,
        model="m",
        seed=seed,
        dataset="emails",
        version=7,
        side=side,
        samples=29,
        metrics={"entities.f1": f1, "entities.partial_f1": partial, "mask.leaked": 0.1},
        per_class=per_class or {"PER": {"f1": f1, "partial_f1": partial}},
    )


def test_trials_differing_only_in_seed_are_one_row_with_a_spread():
    table = rows(
        [
            trial("tagger", 1, 0.70, 0.86),
            trial("tagger", 2, 0.72, 0.84),
            trial("gliner", None, 0.5, 0.8),
        ]
    )
    assert [(r.approach, r.n) for r in table] == [("tagger", 2), ("gliner", 1)]
    mean, spread = table[0].metric("entities.f1")
    assert round(mean, 3) == 0.71 and round(spread, 4) == 0.0141
    assert table[1].metric("entities.f1") == (0.5, None)


def test_a_validation_side_is_labelled_as_dev():
    [row] = rows([trial("tagger", 1, 0.7, 0.86)])
    assert row.set_label == "dev: emails v7 val, 29"
    [held] = rows([trial("tagger", 1, 0.7, 0.86, side="holdout")])
    assert held.set_label.startswith("holdout:")


def test_the_tables_carry_strict_and_partial_overall_and_per_class():
    table = rows([trial("tagger", 1, 0.70, 0.86), trial("tagger", 2, 0.72, 0.84)])
    header, [cells] = headline(table)
    assert header[:5] == ["set", "approach", "n", "F1", "partial F1"]
    assert cells[3] == "0.710 ± 0.014" and cells[4] == "0.850 ± 0.014"
    header, [cells] = by_class(table)
    assert header == ["set", "approach", "PER"] and cells[2] == "0.71 / 0.85"
    [payload] = as_json(table)
    assert payload["seeds"] == [1, 2] and payload["per_class"]["PER"]["partial_f1"]["mean"] == 0.85


def test_report_reads_what_ran_from_the_ledger(experiment_file, config_file, capsys):
    dev = experiment_file.read_text().replace('on = "holdout"', 'on = "val"')
    experiment_file.write_text(dev)
    assert main(["report", str(experiment_file)]) == 0
    assert "Nothing evaluated yet" in capsys.readouterr().out

    assert main(["run", str(experiment_file), "--config", str(config_file)]) == 0
    capsys.readouterr()
    experiment = load(experiment_file)
    found = scored(experiment, ledger_for(experiment))
    assert {t.side for t in found} == {"val"} and len(found) == 2

    assert main(["report", str(experiment_file), "--markdown"]) == 0
    out = capsys.readouterr().out
    assert "| dev: demo v1 val," in out and "lr=0.1" in out and "lr=0.5" in out
    assert main(["--json", "report", str(experiment_file)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
