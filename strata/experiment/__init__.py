"""An experiment, written down once and run from the file.

A project is the durable job; an experiment is a file that references one
and adds variation: an ordered list of stages, each a registered name with
literal arguments, and a grid of values to vary. The file's canonical JSON
is hashed, and a ledger under the project records every stage that ran by
the hash of everything upstream of it. The stages are the other packages':
the orchestrator decides the calls and records what came back. See
``docs/adr/0037``.
"""

from .ledger import Ledger
from .registry import STAGES, ChainError, resolve
from .run import TrialResult, run_experiment
from .spec import Experiment, ExperimentError, StageSpec, Trial, load

__all__ = [
    "STAGES",
    "ChainError",
    "Experiment",
    "ExperimentError",
    "Ledger",
    "StageSpec",
    "Trial",
    "TrialResult",
    "load",
    "resolve",
    "run_experiment",
]
