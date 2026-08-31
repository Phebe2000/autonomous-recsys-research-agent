# Agent A vertical milestone

This track wraps the unchanged official `data.py`, `baseline.py`, `evaluate.py`,
and `submit.py`. It adds input alignment checks, content-addressed dataset
onboarding, a persistent SQLite research ledger, an atomic maximum of 50 trials
per dataset fingerprint, validation-only selection, and a NumPy Soft-target
ListNet candidate.

The prior in `configs/listnet_prior.json` transfers only the predecessor method
and hyperparameters. Every dataset trains a fresh official FM initialization.
No old checkpoint, embedding, artifact, or score is accepted as a prior.

Runtime state is stored under `runtime/<dataset fingerprint>/` and ignored by
Git. Each fingerprint owns its own immutable budget ledger, research-memory
events, checkpoints, and `top1.json`. Development KuaiRand-Pure therefore cannot
consume a future dataset's budget.

Run tests:

```bash
.venv/bin/python -m unittest discover -s tracks/agent_a/tests -v
```

Run the development vertical pipeline on an empty fingerprint ledger:

```bash
.venv/bin/python -m tracks.agent_a.pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

The runner reserves and commits a trial before candidate code executes. Failed,
pruned, interrupted, reserved, and running trials all consume budget. Resuming
an existing trial ID never inserts another row. Trial results accept validation
metrics only; Top-1 orders completed trials by validation primary, with the
lowest ordinal as the deterministic tie break.
