"""An experiment, written down once and run from the file.

A file names a project, an ordered list of stages and a grid; a ledger
records every stage by the hash of everything upstream. See ``docs/adr/0037``.
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
