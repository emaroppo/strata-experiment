"""``strata-experiment`` checks and runs a file, and says what each trial reported."""

import json

import pytest

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
    assert "evaluate     ran" in out and "holdout classify: exact_match=" in out

    assert main(["--json", "run", str(experiment_file), "--config", str(config_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [t["reused"] for t in payload] == [5, 5]
    assert all(t["metrics"]["classify.recall"] > 0.5 for t in payload)


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


def test_the_environment_names_the_config(experiment_file, config_file, capsys, monkeypatch):
    monkeypatch.setenv("STRATA_CONFIG", str(config_file))
    assert main(["--json", "run", str(experiment_file)]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2

    monkeypatch.setenv("STRATA_CONFIG", str(config_file.parent / "absent.toml"))
    assert main(["run", str(experiment_file)]) == 1
    assert "STRATA_CONFIG" in capsys.readouterr().err


def test_a_refusal_is_one_line_on_stderr_and_nothing_on_stdout(experiment_file, capsys):
    broken = experiment_file.read_text().replace('[[stage]]\nuse = "materialise"\n', "")
    experiment_file.write_text(broken)
    assert main(["--json", "check", str(experiment_file)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1 and "needs a dataset_dir" in captured.err


def test_set_without_a_value_is_a_refusal(experiment_file, capsys):
    assert main(["check", str(experiment_file), "--set", "train.params.lr"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "--set takes KEY=VALUE, got 'train.params.lr'\n"


def test_a_value_error_in_the_body_is_a_crash(experiment_file, monkeypatch):
    def broken(pairs):
        raise ValueError("a bug, not a refusal")

    monkeypatch.setattr("strata.experiment.cli._overrides", broken)
    with pytest.raises(ValueError, match="a bug"):
        main(["check", str(experiment_file)])


def test_json_check_is_the_record_whole(experiment_file, capsys):
    assert main(["--json", "check", str(experiment_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"id", "name", "stages", "trials"}
    assert payload["name"] == "lr"
    assert all(set(t) == {"id", "overrides"} for t in payload["trials"])


def test_run_hands_the_stages_something_to_report_progress_to(
    experiment_file, config_file, monkeypatch, capsys
):
    """The catalog's fetching, a model's epochs and a remote job's state each reach the terminal
    through the handles (``docs/adr/0031``).
    """
    handed = {}

    def fake(experiment, handles, ledger=None, on_event=None):
        handed["handles"] = handles
        return []

    monkeypatch.setattr("strata.experiment.cli.run_experiment", fake)
    assert main(["--json", "run", str(experiment_file), "--config", str(config_file)]) == 0
    assert json.loads(capsys.readouterr().out) == []
    handles = handed["handles"]
    assert handles.on_progress is not None
    assert handles.on_state is not None
    assert handles.on_epoch is not None


def test_the_ticker_draws_on_a_terminal_only(monkeypatch, capsys):
    from strata.experiment.cli import _ticker

    show, done = _ticker("running")
    show("materialising 1/2")
    done()
    assert capsys.readouterr() == ("", "")

    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    show, done = _ticker("running")
    show("materialising 1/2")
    show("epoch 1/3")
    done()
    done()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "\rrunning materialising 1/2\rrunning epoch 1/3\n"
