# Agent AI-yoh: An Auditable Autonomous Recommender-System Research Agent

## Elevator pitch

Agent AI-yoh turns recommender-system experimentation into a bounded, reproducible,
and reviewable research loop. Given a fixed dataset, evaluator, and compute
budget, it proposes model candidates, trains them, evaluates only on validation,
records every decision, stops under a predeclared convergence rule, and emits a
single label-free score for every logged test exposure.

## The problem

The challenge asks participants to improve the KuaiRand-Pure logged-exposure
ranking baseline while demonstrating genuine ML-research autonomy. A useful
solution must do more than train one model: it must respect temporal splits,
avoid hidden-test leakage, manage a strict 50-iteration and six-hour budget,
preserve hypotheses and code changes, recover from failures, and select the
final checkpoint using validation evidence only.

## Our solution

Agent AI-yoh implements the complete research lifecycle as software rather than as an
informal notebook process.

1. A label-safe data boundary exposes train labels for fitting, validation labels
   only to the official evaluator, and test rows as unlabeled exposures.
2. A content fingerprint gives every dataset an independent ledger, Optuna
   study, budget, artifacts, Top-1, and lock.
3. A strict `CandidateConfig` unifies FM, Soft-target ListNet, causal positive
   History, BPR, multi-task heads, LightGBM LambdaRank, and FM+LambdaRank
   ensembles while rejecting contradictory combinations.
4. `ResearchStore` atomically reserves budget before training. Failed, pruned,
   interrupted, and running trials remain counted; exact duplicates may reuse
   completed evidence without a new charge.
5. The reward and selection key are always validation primary, defined as the
   mean of official GAUC and nDCG@5. Test metrics are rejected at every layer.
6. Every research decision enters an append-only SHA-256 hash chain containing
   its rationale, evidence, alternatives, selected action, trial, and data scope.
7. At convergence, Agent AI-yoh locks the stable validation-best checkpoint and
   produces one finite, row-aligned score per official test exposure without
   loading `long_view` or calculating hidden metrics.

## Model design

The six controlled anchors compared the official FM, Soft-target ListNet,
strictly causal positive-history mean pooling, causal-behavioral LambdaRank, an
FM+LambdaRank ensemble, and a wider-tree ranker.

The LambdaRank candidate groups all training exposures by user and directly
optimizes logged-exposure ranking. It uses the five official categorical fields
plus duration/time features and smoothed video, author, user-video, user-author,
and user-duration statistics. For each training row, target-derived aggregates
use only strictly earlier timestamps; tied, current, and future rows are
excluded. Validation and test features are frozen from train.

Standalone LambdaRank underperformed FM, but the two models provided weakly
complementary rankings. The final trial uses within-user normalized scores:

```text
score = 0.35 * z_user(LambdaRank) + 0.65 * z_user(FM)
```

## Autonomy and auditability

The run began with six predeclared anchors. Evidence from A-05 motivated a local
blend screen. The team made one manual research intervention to direct
that screen; all other candidate execution, validation, Top-1 promotion,
convergence, locking, and final inference decisions were recorded by Agent AI-yoh.

The run stopped after 9/50 iterations under the frozen cumulative convergence
rule (`epsilon=0.002`, `N=3`, minimum scored iterations=9). The audit package
contains per-iteration hypotheses, exact code-diff digests, metrics, lifecycle
events, the 21-decision hash chain, immutable lock, checkpoint digest, resource
usage, and submission digest.

## Results

The locked trial achieved validation GAUC `0.66910648`, nDCG@5 `0.53661937`, and
primary `0.60286292`. Against the organizer's rounded validation baseline
(`0.6674`, `0.5357`, `0.6016`), the approximate deltas are `+0.00170648`,
`+0.00091937`, and `+0.00126292` respectively. Because the primary gain is below
the declared `epsilon=0.002`, we report it as slightly positive, not significant.
The hidden score is unknown and will be produced only by the organizer.

## Development tools

- OpenAI Codex desktop coding environment and coding agent
- macOS Terminal / zsh
- Git and GitHub
- Python virtual environment
- Python `unittest` with warnings treated as errors
- SQLite for persistent research state

VS Code, Colab, and Jupyter were not required for the submitted workflow.

## APIs and model services

- OpenAI Codex service was used for development and research-agent interaction.
- The autonomous runner supports a read-only `gpt-5.6-sol` Codex coding-provider
  phase, but the final judged run converged during controlled screening before
  an in-run provider candidate was invoked.
- No third-party data API was used.

## Libraries and frameworks

- NumPy 2.5.2
- LightGBM 4.7.0
- Optuna 4.9.0 with seeded `TPESampler`
- SciPy 1.18.1
- narwhals 2.25.0, installed as an isolated Agent dependency
- Python standard-library `sqlite3`, `csv`, `hashlib`, and `unittest`

The official evaluator and baseline remain organizer-provided NumPy code.

## Datasets and assets

- Required dataset: KuaiRand-Pure only
- Train: 1,141,112 rows, dates 2022-04-08 through 2022-04-21
- Validation: 124,909 rows, dates 2022-04-22 through 2022-04-28
- Test inference: 170,588 rows, dates 2022-04-29 through 2022-05-08
- Official Starter Kit files and static video metadata

No external training data, random-exposure training rows, manually labeled
examples, KuaiRand-1K/27K transfer, or pretrained benchmark weights were used.

## Limitations and next steps

The validation improvement is below the challenge's default significance
epsilon, the judged run used seed 0, and hidden performance is not yet known.
The standalone ranker was weaker than FM, and the final ensemble gain is modest.
With more time, we would validate the method on an independent KuaiRand-1K
fingerprint, improve feature caching and memory use, and predeclare a compact
cross-seed finalist verification policy. We would not use bonus data to train or
select the Pure model.

## Team

**Team name:** AI-yoh!

**Team members:** Wang Wei Yu, NI MAN-LING, Lee Hsin-Jui, and Huang Yuan-Heng.

The team jointly developed and reviewed the project direction, validation
evidence, compliance boundaries, and final submission. Agent AI-yoh implemented,
tested, executed, audited, and documented the autonomous research pipeline. The
append-only run logs distinguish automated agent actions from the team's one
recorded manual research intervention.
