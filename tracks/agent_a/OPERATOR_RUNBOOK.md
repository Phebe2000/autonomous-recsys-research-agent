# KuaiRand-Pure judged-run operator runbook

This is the authoritative procedure for the required benchmark. It does not
authorize hidden-label access or hidden-test scoring.

## 1. Preflight

From the repository root:

```bash
tracks/agent_a/check.sh
.venv/bin/python -m tracks.agent_a.judged_cli plan \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime
```

Confirm the official four files are unchanged; task is `long_view` within-user
logged-exposure ranking; metrics are GAUC/nDCG@5; budget is 0/50; and output says
`hidden_labels_loaded=false`, `test_metrics_used=false`. Plan is read-only.

The safe fingerprint includes all permitted train fields, public validation
`long_view`, hidden-test inference fields, and static feature files. It excludes
hidden-test feedback columns and the random-exposure log. Changing hidden labels
cannot alter fingerprint, model state, selection, or final scores.

## 2. Freeze policy and create 0/50

Policy values must be fixed before iteration 1:

```bash
.venv/bin/python -m tracks.agent_a.judged_cli init \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime \
  --epsilon 0.002 \
  --convergence-n 3 \
  --minimum-scored-iterations 9
```

`init` writes the immutable policy, initial code snapshot, dataset manifest,
resource counters, audit log, and empty ResearchStore. It does not create an
Optuna study, reserve a trial, or start the six-hour clock. Do not delete this
directory to regain budget.

If—and only if—an earlier preflight has zero reserved trials and its clock has
never started, a code change before iteration 1 may be handled with
`judged_cli supersede-empty`. The command moves the complete old preflight into
`superseded/` with a reason and digest, then requires a fresh `init`. It refuses
any ledger that has spent budget or started its clock. Never remove or rewrite
the superseded evidence.

## 3. Start and resume

```bash
.venv/bin/python -m tracks.agent_a.judged_cli run \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime \
  --provider-model gpt-5.6-sol
```

The first real reservation starts the persistent six-hour clock. The first six
iterations are controlled anchors: official FM, Soft-target ListNet, causal
last-20 History, causal-behavioral LambdaRank, FM+LambdaRank within-user
normalized ensemble, and a wider-leaf LambdaRank ablation.
All initialize from this benchmark only. Random-log rows and 1k/27k artifacts are
forbidden as Pure training input.

LambdaRank features include direct time/duration fields and smoothed video,
author, user-video, user-author and user-duration behavioral statistics. Every
training target statistic is computed from strictly earlier timestamps; current,
tied and future labels are excluded. Validation and final inference aggregates
are frozen from train. The ranker checkpoint records its feature-schema digest,
LightGBM model text, selected validation iteration and any FM blend weight.

From iteration 7 onward, odd-numbered trials use the unified conditional
screening/TPE path for modules with positive fingerprint-local evidence;
even-numbered trials use the coding provider. The provider receives only
validation-safe research evidence and proposes a self-contained NumPy candidate. Each proposal
must state a hypothesis, rationale, expected metric effect, reflection plan and
resource estimate. The provider runs read-only: it cannot edit the repository.
The harness validates the returned abstract syntax tree and function contract,
writes the exact generated source and unified diff, then executes the candidate
through the guarded trainer. The candidate may use the five encoded categorical
fields plus safe date/hour/time/duration side features. Click and play-time are
training-only auxiliary targets; validation and final inference never expose
them. Filesystem, network, subprocess, environment, dynamic import and test-data
access are rejected.

After validation (or an execution error), a separate provider call records a
structured reflection and recommended next action. That evidence, previous
metrics and recovery events become context for the next proposal. Provider
infrastructure failure consumes the already-reserved iteration and fatal-stops
the loop, avoiding repeated budget loss during an outage. A rejected proposal or
failed candidate also remains counted. Generated candidates do not bypass the
same validation-only reward, convergence policy, clock or stable Top-1 rule.

The ledger reserves before training. Completed, failed, pruned, running,
reserved, interrupted, and deadline-crossing iterations count toward 50. A
duplicate canonical configuration may reuse a result without another reservation.

After interruption, preserve every file and resume:

```bash
.venv/bin/python -m tracks.agent_a.judged_cli inspect \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime
.venv/bin/python -m tracks.agent_a.judged_cli resume \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime \
  --provider-model gpt-5.6-sol
```

Resume retains the original deadline, budget, policy, Optuna history, optimizer,
heads, and checkpoints. Never edit SQLite or status fields manually.

## 4. Convergence and audit evidence

With default policy, convergence occurs after the minimum floor when:

```text
best(last 3 scored iterations) - best(all earlier scored iterations) <= 0.002
```

Crashes without validation scores count against budget/time but neither advance
nor reset the scored window. Other stop reasons are 50/50, six-hour deadline,
fatal guard, or explicit operator lock.

Each iteration audit record contains the complete research action, hypothesis,
rationale, expected effect, exact source/diff and their SHA-256 digests, provider
and model, raw provider event-log path, contract-validation decision, validation
metrics or error, structured reflection, lifecycle/recovery events, config
identity, timestamps and measured token usage. Configuration-only anchors carry
an explicit empty diff. Report any additional resource usage truthfully:

All agent choices are recorded through the single
`ResearchStore.record_agent_decision(...)` contract. The SQLite event is the
append-only authority; `agent_decision_journal.json` is rebuilt and hash-chain
verified during every audit write. A decision records stage, actor, rationale,
validation-safe evidence, alternatives, selected action, trial and data scope.
Idempotency keys prevent resume from duplicating a prior choice. Finalization
adds a `label_free_locked_inference` decision containing checkpoint and output
digests, but never labels or metrics. Preserve this journal with the run logs
and final submission manifest for organizer review.

```bash
.venv/bin/python -m tracks.agent_a.judged_cli record-usage \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime \
  --llm-input-tokens 0 --llm-output-tokens 0 \
  --gpu-seconds 0 --manual-interventions 0
```

Replace zero increments with measured values. Never infer or fabricate past code
diffs, token counts, GPU time, or interventions.

## 5. Validation lock and final submission

At stop, inspect and lock the stable validation-best checkpoint:

```bash
.venv/bin/python -m tracks.agent_a.judged_cli inspect \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime
.venv/bin/python -m tracks.agent_a.judged_cli lock \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime
```

Selection is validation primary, with lowest trial ordinal as tie-break. Hidden
test or leaderboard information must never change the lock.

```bash
.venv/bin/python -m tracks.agent_a.finalize submission.csv \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime \
  --split test --judged
.venv/bin/python -m tracks.agent_a.safe_submit_check submission.csv \
  --data-dir KuaiRand-Pure/data --split test
```

Finalization reads test exposure/features only, verifies fingerprint, lock and
checkpoint digest, and emits one finite score per official row. It does not load
`long_view`, import the evaluator, or compute a hidden metric. Submit this locked
file once. Organizer model-effect score is `hidden primary - 0.5946`.

## Final checklist

- [ ] Official files and evaluator hashes pass.
- [ ] Judged policy and initial code snapshot were frozen before iteration 1.
- [ ] Ledger started at 0/50 and was never reset.
- [ ] Persistent clock started at first reservation and remains within six hours.
- [ ] Only train labels trained the model; validation only evaluated/selected.
- [ ] Test-label mutation non-interference and random-log exclusion tests pass.
- [ ] Pure used no 1k/27k data, weights, embeddings, or checkpoints.
- [ ] Failed/pruned/crashed iterations remain counted and absent from scored window.
- [ ] Iterations 1–6 are controlled anchors; later iterations use the guarded coding loop.
- [ ] Every iteration has hypothesis, exact diff, metrics/error, events and timestamps.
- [ ] Generated source passed AST/contract checks and its recorded hashes reproduce.
- [ ] Provider/model/raw logs, reflection and recovery events are preserved.
- [ ] Decision journal hash chain verifies through final lock and score emission.
- [ ] Token, GPU and manual-intervention totals are truthful.
- [ ] Final lock is validation-best and immutable.
- [ ] Submission passes label-free schema/alignment check.
- [ ] Hidden test is evaluated once and never feeds back into research.
