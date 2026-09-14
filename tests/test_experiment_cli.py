"""``strata-experiment`` checks and runs a file, and says what each trial reported."""

import json

from strata.experiment.cli import main


def test_check_lists_the_trials(experiment_file, capsys):
    assert main(["check", str(experiment_file)]) == 0
    out = capsys.readouterr().out
    assert "trials   2" in out and "train.params.lr=0.5" in out
    assert main(["--json", "check", str(experiment_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stages"] == ["dataset", "materialise", "split", "train", "evaluate"]
    assert len(payload["trials"]) == 2


def test_check_refuses_a_broken_chain(experiment_file, capsys):
    broken = experiment_file.read_text().replace('[[stage]]\nuse = "materialise"\n', "")
    experiment_file.write_text(broken)
    assert main(["check", str(experiment_file)]) == 1
    assert "needs a dataset_dir" in capsys.readouterr().err


def test_run_reports_each_trial(experiment_file, config_file, capsys):
    assert main(["run", str(experiment_file), "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert out.count("trial ") == 2
    assert "evaluate     ran" in out and "exact_match=" in out

    assert main(["--json", "run", str(experiment_file), "--config", str(config_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [t["reused"] for t in payload] == [5, 5]
    assert all(t["metrics"]["recall"] > 0.5 for t in payload)


def test_set_overrides_the_file(experiment_file, config_file, capsys):
    code = main(
        [
            "--json",
            "run",
            str(experiment_file),
            "--config",
            str(config_file),
            "--set",
            "train.params.lr=0.9",
        ]
    )
    assert code == 0
    [trial] = json.loads(capsys.readouterr().out)
    assert trial["records"][3]["request"]["params"] == {"lr": 0.9}
