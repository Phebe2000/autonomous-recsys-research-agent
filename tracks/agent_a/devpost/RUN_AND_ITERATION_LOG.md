# KuaiRand-Pure judged-run log

## Run policy

- Dataset fingerprint: `sha256:73967e0b7ff70aee8d25dc3832bad5de2a7a3cf4a8130f20dcaa93513f92e4f6`
- Reward and selection: validation primary only
- Stable tie-break: lowest trial ordinal
- `epsilon=0.002`, `N=3`, minimum scored iterations=9
- Hard caps: 50 iterations and six hours
- Start: `2026-08-31T18:39:14.165424Z`
- Recorded convergence stop: `2026-08-31T19:03:11.836714Z`
- Agent wall-clock to recorded stop: 1,437.671 seconds (23m 57.671s)
- Test labels loaded: false
- Test metrics used: false

## Code provenance

Trials 1–6 executed preimplemented anchors from commit
[`c1bf36f`](https://github.com/Phebe2000/autonomous-recsys-research-agent/commit/c1bf36f).
Their per-iteration code diff is intentionally empty because the code and
configs were frozen before the run; the empty-diff SHA-256 is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Trials 7–9 used the audited A-05 screening change later preserved in commit
[`ae38beb5b78a3df00f6d9a3df314de41e09951d1`](https://github.com/Phebe2000/autonomous-recsys-research-agent/commit/ae38beb5b78a3df00f6d9a3df314de41e09951d1).
The exact unified diff is embedded in the machine-readable audit log; its
SHA-256 is `d93a74256860128060740449c4844936e7e69378f6368d0bfc0178d73d22825f`.

## Iterations

### Trial 01 — Official FM reproduction

- Hypothesis: establish the organizer FM reference under the exact judged data,
  evaluator, seed, and checkpoint-selection contract.
- Change: preimplemented anchor; empty in-run diff.
- Result: GAUC `0.66713339`, nDCG@5 `0.53580570`, primary `0.60146955`.
- Best step: epoch 7.
- Error/recovery: none.

### Trial 02 — Soft-target ListNet

- Hypothesis: aligning the training loss with within-user ranking may improve
  validation primary over pointwise FM.
- Change: preimplemented ListNet anchor; empty in-run diff.
- Result: GAUC `0.66713190`, nDCG@5 `0.53581190`, primary `0.60147190`.
- Best step: 40.
- Error/recovery: none. The gain was effectively zero.

### Trial 03 — Causal positive-history mean pooling

- Hypothesis: strictly earlier positive-video history with a fixed residual gate
  may add short-term preference signal without validation leakage.
- Change: preimplemented History anchor; empty in-run diff.
- Result: GAUC `0.66718662`, nDCG@5 `0.53583813`, primary `0.60151237`.
- Best step: 30.
- Error/recovery: none. Improvement remained far below epsilon.

### Trial 04 — Causal-behavioral LambdaRank

- Hypothesis: user-grouped LambdaRank with causal behavioral aggregates may
  outperform FM by directly optimizing logged-exposure ranking.
- Change: preimplemented ranker anchor; empty in-run diff.
- Result: GAUC `0.66237837`, nDCG@5 `0.53349775`, primary `0.59793806`.
- Best tree iteration: 80.
- Error/recovery: no execution error. Evidence rejected standalone ranker
  promotion because primary fell `0.00353149` below trial 01.

### Trial 05 — FM + LambdaRank ensemble

- Hypothesis: although standalone LambdaRank is weaker, its ranking errors may be
  complementary to FM after within-user normalization.
- Change: preimplemented ensemble anchor; empty in-run diff.
- Result: GAUC `0.66889727`, nDCG@5 `0.53655154`, primary `0.60272440`.
- Best tree iteration / ranker weight: 160 / 0.50.
- Error/recovery: none. Promoted to validation Top-1.

### Trial 06 — Wider-leaf LambdaRank

- Hypothesis: a wider tree may capture higher-order interactions missed by the
  31-leaf ranker.
- Change: preimplemented wider-ranker anchor; empty in-run diff.
- Result: GAUC `0.66231507`, nDCG@5 `0.53374135`, primary `0.59802821`.
- Best tree iteration: 40.
- Error/recovery: no execution error. Evidence rejected wider standalone ranker.

### Trial 07 — Ensemble blend 0.35

- Hypothesis: A-05 suggests complementarity, but the weaker ranker should receive
  less than 0.50 weight.
- Change: configuration-only A-05 local screen; diff SHA
  `d93a74256860128060740449c4844936e7e69378f6368d0bfc0178d73d22825f`.
- Config: ranker weight 0.35, 160 trees, 31 leaves.
- Result: GAUC `0.66910648`, nDCG@5 `0.53661937`, primary `0.60286292`.
- Best tree iteration: 160.
- Error/recovery: none. Promoted to validation Top-1 and later locked.

### Trial 08 — Ensemble blend 0.45

- Hypothesis: test an intermediate blend between trial 07 and A-05 to determine
  whether the optimum is locally smooth.
- Change: configuration-only A-05 local screen; same audited diff SHA.
- Config: ranker weight 0.45, 160 trees, 31 leaves.
- Result: GAUC `0.66874242`, nDCG@5 `0.53644580`, primary `0.60259411`.
- Best tree iteration: 160.
- Error/recovery: none. Retained as ablation evidence; trial 07 remained Top-1.

### Trial 09 — Ensemble blend 0.55

- Hypothesis: test whether greater ranker influence can recover additional GAUC
  or nDCG despite standalone underperformance.
- Change: configuration-only A-05 local screen; same audited diff SHA.
- Config: ranker weight 0.55, 160 trees, 31 leaves.
- Result: GAUC `0.66824031`, nDCG@5 `0.53644705`, primary `0.60234368`.
- Best tree iteration: 140.
- Error/recovery: none. Retained as negative local evidence.

## Stop and recovery events

No model-training iteration failed, crashed, or was pruned. After trial 09, the
predeclared cumulative rule detected no improvement greater than 0.002 in the
last three scored iterations relative to earlier evidence. Agent AI-yoh recorded
`converged_no_0.002_gain_for_3_scored_iterations` and refused to spend trials
10–14.

A post-run audit issue was discovered: later report generation initially
recomputed elapsed time from the current clock, even though the run had already
stopped. The repair derives immutable `stopped_at` from the append-only stopping
decision, freezes elapsed time at 1,437.671 seconds, and does not alter any trial,
metric, checkpoint, budget, or model selection.

## Manual-intervention summary

Manual research interventions: **1**.

After the six controlled anchors, the team explicitly directed Agent AI-yoh
to concentrate screening on A-05 blend weight, tree iterations, and leaf count.
Starting the run, requesting status, locking the already selected Top-1, and
producing the final score file are treated as routine operator actions rather
than research-direction interventions.
