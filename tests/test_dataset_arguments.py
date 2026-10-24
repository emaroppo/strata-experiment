"""What a dataset stage in an experiment file may say about the version it freezes."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from strata.catalog.stages import DatasetRequest
from strata.experiment import ChainError
from strata.experiment.registry import check
from strata.experiment.run import _request
from strata.experiment.spec import from_payload

#: As much of a project as a dataset stage's request reads.
PROJECT = SimpleNamespace(
    catalog=SimpleNamespace(group_by="", given_split=None),
    label_set_name="verdict",
    collections=["phishnet/sft"],
    dataset_name="sft",
    root=None,
)


def _experiment(**dataset):
    return from_payload(
        {
            "project": "demo",
            "name": "sft",
            "stage": [{"use": "dataset", **dataset}, {"use": "materialise"}],
        },
        project_id="demo",
    )


def _requested(experiment):
    handles = cast(Any, SimpleNamespace(project=PROJECT))
    return _request(experiment.stages[0], handles, {}, "e1")


def test_the_dataset_stage_says_what_to_do_with_disagreeing_inputs():
    """A freeze over fine-tuning examples leaves them out or keeps the first. docs/adr/0053"""
    experiment = _experiment(disagreeing_inputs="first")
    check(experiment)

    request = _requested(experiment)
    assert isinstance(request, DatasetRequest)
    assert request.disagreeing_inputs == "first"
    default = _requested(_experiment())
    assert isinstance(default, DatasetRequest) and default.disagreeing_inputs == "drop"


def test_another_value_or_a_misspelt_key_is_refused_before_the_stage_runs():
    with pytest.raises(ValueError, match="drop"):
        _requested(_experiment(disagreeing_inputs="last"))
    with pytest.raises(ChainError, match="disagreeing_input"):
        check(_experiment(disagreeing_input="first"))
