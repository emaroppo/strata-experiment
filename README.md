# strata-experiment

An experiment written down once: an ordered list of stages from a file, a
grid of values to vary over them, and a ledger of what ran, so a stage
already run for the same inputs is handed downstream rather than run
again. The top of the strata graph: it sequences every other package's
stages and nothing imports it.

```bash
uv add strata-experiment
```

Depends on `strata-labels`, `strata-common`, `strata-catalog`,
`strata-modelling` and `strata-labeller`. May not know Label Studio exists.

## An experiment is a file

```toml
project = "cats-dogs"          # a project directory: catalog, label set, model

[[stage]]
use = "dataset"                # freeze what is labelled into a version
val_ratio = 0.2
holdout_ratio = 0.1            # kept back; nothing below trains or selects on it

[[stage]]
use = "materialise"            # the version on disk, files by checksum

[[stage]]
use = "split"                  # the version's sides, or a drawn split when unlocked

[[stage]]
use = "train"
fresh = true                   # every trial cold, so they compare

[[stage]]
use = "evaluate"
on = "holdout"                 # one implementation of the score, for every model

[grid]
"train.params.lr" = [0.003, 0.001, 0.0003]
```

```bash
strata-experiment check experiment.toml     # validate, check the chain, list the trials
strata-experiment run   experiment.toml     # run them in order, reusing what the ledger holds
strata-experiment run   experiment.toml --set train.params.lr=0.01   # override before hashing
```

`--json` prints the records instead of rendering them.

## How it is reproducible

**The file hashes, and the hash is the identity.** TOML is what a person
edits; its canonical JSON is what is hashed. A trial is the file with its
overrides written in. A stage's key is the hash of the spec through that
stage plus the version of the implementation that ran it, computable
before anything runs.

**A stage is a function** taking a request and a context and returning a
record that names what it made, a dataset version or a run, by identity.
The stages are the same functions the command line calls one at a time,
registered in one table: `dataset`, `materialise` and `split` from the
catalog, `train` and `evaluate` from modelling.

**The ledger makes a rerun cheap.** Under the project, every record by its
key, and a directory per trial holding a copy of each record it used. A
stage whose key already has a record is handed downstream as if it had
run, so trials that agree through a stage share it, a change reruns only
what is downstream of it, and a rerun of an unchanged file does nothing.

**The chain is checked before the first stage runs.** A stage before what
it needs, or a grid whose trials would warm-start from each other, is
refused at load. Selection reads validation; the result is reported on a
holdout the study never sees. By default a trial reuses the sides the
version carries; a seed on the `split` stage draws its own, and the draw
is recorded either way.

## Decisions

The design in full is `docs/orchestrator.md` in the umbrella repository,
with the frozen version and its holdout in `docs/adr/0003`.

## Tests

```bash
uv run pytest packages/experiment
```

The suite runs against a toy model and a temporary catalog; nothing needs
a GPU.
