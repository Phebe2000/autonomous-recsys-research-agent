# Agent A autonomous recommender research

## 中文摘要

Agent A 保留官方 `data.py`、`baseline.py`、`evaluate.py`、`submit.py`，以
`long_view`、GAUC、nDCG@5 與 validation primary 完成 KuaiRand-Pure 研究。
正式 judged-run 使用 label-inaccessible data boundary：train/validation 可以
取得標籤，test 只提供 exposure 與 inference features。每輪 failed、pruned、
interrupted 皆計入 50 次，並受 persistent 6 小時 wall-clock 限制。

既有 24-trial ledger 是 development evidence，不是正式 judged run。正式 run
使用獨立 fingerprint、0/50 ledger、immutable convergence policy、Optuna study、
iteration audit log 與 validation-only final lock。操作程序以
[OPERATOR_RUNBOOK.md](OPERATOR_RUNBOOK.md) 為準。

## English summary

Agent A preserves the official starter kit and targets KuaiRand-Pure long-view
within-user ranking with GAUC, nDCG@5, and validation primary. The judged-run
boundary exposes labels for train/public validation only; hidden-test rows expose
inference fields but no relevance labels. Every terminal or interrupted attempt
counts toward 50, and a persistent six-hour deadline applies across resume.

The existing 24-trial ledger remains development evidence. A judged run owns an
independent safe fingerprint, clean 0/50 ledger, immutable convergence policy,
persistent Optuna study, per-iteration audit log, and validation-only final lock.

## Canonical judged-run commands

```bash
.venv/bin/python -m pip install -r tracks/agent_a/requirements.txt
# macOS LightGBM runtime: brew install libomp
tracks/agent_a/check.sh

# Read-only; does not create a ledger or start the clock.
.venv/bin/python -m tracks.agent_a.judged_cli plan \
  --data-dir KuaiRand-Pure/data

# Freeze ε/N/minimum floor and create the clean 0/50 ledger. Clock remains stopped.
.venv/bin/python -m tracks.agent_a.judged_cli init \
  --data-dir KuaiRand-Pure/data \
  --epsilon 0.002 --convergence-n 3 --minimum-scored-iterations 9

# First reservation starts the persistent six-hour clock. Iterations 1-6 are
# controlled anchors; later trials alternate conditional unified refinement and
# the Codex code-generation provider.
.venv/bin/python -m tracks.agent_a.judged_cli run \
  --data-dir KuaiRand-Pure/data --provider-model gpt-5.6-sol
.venv/bin/python -m tracks.agent_a.judged_cli resume \
  --data-dir KuaiRand-Pure/data --provider-model gpt-5.6-sol
.venv/bin/python -m tracks.agent_a.judged_cli inspect --data-dir KuaiRand-Pure/data
.venv/bin/python -m tracks.agent_a.judged_cli lock --data-dir KuaiRand-Pure/data

# Produce and validate final exposure scores without loading hidden labels.
.venv/bin/python -m tracks.agent_a.finalize submission.csv \
  --data-dir KuaiRand-Pure/data \
  --state-root tracks/agent_a/judged_runtime --split test --judged
.venv/bin/python -m tracks.agent_a.safe_submit_check submission.csv \
  --data-dir KuaiRand-Pure/data --split test
```

After the six anchors, odd-numbered trials use conditional unified/Optuna
refinement while even-numbered trials use the installed Codex CLI read-only to return a
schema-constrained research proposal and NumPy source module. Agent A validates
the AST, records and materializes the exact unified diff, trains it against a
label-safe context, evaluates with the official validation function, and asks a
separate reflection turn to guide the next proposal. Generated code can engineer
train-only click/play-time signals and common date/time/duration inference
features, but cannot access files, test feedback, network, subprocesses, or the
compliance implementation.

The six controlled anchors are official FM, Soft-target ListNet, causal
last-20 History, causal-behavioral LambdaRank, an FM+LambdaRank within-user
normalized ensemble, and a wider-leaf LambdaRank ablation. Behavioral target
rates for training rows use strictly-earlier `time_ms` and exclude tied/current
rows; validation/final inference use frozen train-only aggregates. BPR and the
original auxiliary heads remain available as historical ablations but are not
spent as default judged anchors because the current fingerprint showed no gain.

The run log records hypothesis, actual code diff, before/after source hashes,
provider/model, validation checks, metrics, decision, reflection, errors,
recovery events, timing, GPU seconds, LLM tokens, and manual interventions.
Every agent choice also goes through `ResearchStore.record_agent_decision(...)`:
candidate execution/reuse/rejection, Top-1 disposition, stopping, final lock and
label-free score emission form an idempotent SHA-256 chain exported as
`agent_decision_journal.json`. Each entry states rationale, evidence,
alternatives, selected action, actor, data scope and associated trial. Test
metrics are rejected before a decision can enter the journal.
Usage increments are recorded automatically for Codex calls and explicitly with
`judged_cli record-usage` for external usage. Hidden-test evaluation is external
and occurs once after the immutable validation-best lock.

Simulation remains isolated under an explicit synthetic fingerprint:

```bash
.venv/bin/python -m tracks.agent_a.autonomous_cli simulate --max-trials 50
```
