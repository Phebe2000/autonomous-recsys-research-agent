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

## Milestone 2: causal positive-history mean pooling

`history.py` reads raw `time_ms` as a sidecar without changing the official
loader. A training row can see only same-user positive videos with a strictly
earlier timestamp; current, future, and tied-timestamp rows are excluded.
Validation receives one frozen train-positive snapshot and never consumes
validation feedback. Test histories are unavailable by contract.

The history feature is last-N mean pooling, not an order-preserving sequence
encoder. Controlled windows are 20, 50, 100, and all positives; cold-start rows
receive an exact zero vector. A separate history embedding table adds
`gate * dot(candidate, mean_history) / sqrt(history_dim)` to the unchanged FM
score. Setting the gate to zero takes a bitwise-parity fast path.

Run/reuse the controlled experiments:

```bash
.venv/bin/python -m tracks.agent_a.history_pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

H-00 aliases the existing no-history ListNet trial at zero budget cost. H-01 to
H-04 are exact-config claimed trials. Completed runs are reused; running runs
load their trial-specific checkpoint; failed/pruned runs remain terminal unless
a future explicit retry configuration reserves another trial.

## Same-user BPR weight control

`bpr.py` adds an optional ranking regularizer without changing the no-history FM
scorer. For every discriminative same-user slate it averages all positive versus
negative pair losses, then optimizes `ListNet + bpr_weight * BPR`. Weight zero
takes the exact Milestone 1 ListNet path. Pairs never cross users.

```bash
.venv/bin/python -m tracks.agent_a.bpr_pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

B-00 aliases trial-04 for free; B-01, B-02, and B-03 fix every other setting and
use weights 0.01, 0.05, and 0.10. Selection and early stopping remain validation
only, with test metrics forbidden by the shared trial contract.

## History gate diagnostics

`gate_pipeline.py` keeps the pure Soft-target ListNet objective and fixes causal
last-20 mean history. G-01 starts the gate at zero and freezes the independent
history table; G-02 uses the same gate-only first 50 steps, then permits history
embedding updates with a separate learning rate. The FM/ListNet learning rate
and all validation-only selection rules stay unchanged.

```bash
.venv/bin/python -m tracks.agent_a.gate_pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

## Fixed negative gate control

`fixed_gate_pipeline.py` evaluates fixed gates -0.02, -0.05, and -0.10 with
both the gate and independent history table frozen. Only the baseline-initialized
FM is optimized by the unchanged pure Soft-target ListNet loss. F-00 reuses G-01
as the zero-cost learned-gate reference; selection remains validation-only.

```bash
.venv/bin/python -m tracks.agent_a.fixed_gate_pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

For paired training-order replication, `seed_replication_pipeline.py` runs a
gate-zero control and fixed -0.05 history candidate at seeds 1 and 2. The pair
shares the same fresh current-dataset baseline initialization, so its only model
difference is the history gate; the seed controls user-slate shuffle order.

## Milestone 4: training-only multi-task auxiliaries

`auxiliary.py` aligns raw training rows to the official and encoded row order
using `(date, user_id, video_id, time_ms)` identities. It exposes `is_click` and
a train-fitted `log1p`/z-score play-time target only for training. Validation and
test auxiliary-label access is rejected.

`multitask_model.py` adds task-specific click and play-time heads over the sum of
shared FM embeddings. Auxiliary gradients update the shared embeddings, while
inference still emits exactly one long-view ranking score per row. The controlled
M-01 through M-04 runs keep the primary Soft-target ListNet objective and select
checkpoints only by official validation primary:

```bash
.venv/bin/python -m tracks.agent_a.multitask_pipeline \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```

## Milestone 5: unified candidates and evidence

`candidate.py` defines the strict, serializable candidate contract. Its identity
hash covers the dataset content fingerprint, canonical training configuration,
candidate schema, and code version. Disabled modules are distinct from enabled
modules with explicit zero weights, and unsupported History/BPR/Multi-task
combinations are rejected.

`unified_runner.py` dispatches to the existing baseline, Listwise, History, BPR,
and Multi-task trainer entrypoints without duplicating their mathematics. It
reuses exact terminal configurations before reservation and emits one guarded
validation-only result schema.

The inspect command never authorizes training or reserves a trial. It rebuilds
the evidence registry and JSON/Markdown ablation reports directly from the
fingerprint ledger:

```bash
.venv/bin/python -m tracks.agent_a.unified_inspect \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/runtime
```
