# Final results and resource usage

## Validation results

The organizer's published validation baseline values are rounded to four
decimals. Deltas below are therefore approximate.

| Benchmark | Model | GAUC | Delta vs official | nDCG@5 | Delta vs official | Primary | Delta vs official |
|---|---|---:|---:|---:|---:|---:|---:|
| KuaiRand-Pure official validation baseline | FM | 0.6674 | — | 0.5357 | — | 0.6016 | — |
| KuaiRand-Pure validation-best | FM + LambdaRank, ranker weight 0.35 | 0.66910648 | +0.00170648 | 0.53661937 | +0.00091937 | 0.60286292 | +0.00126292 |

The primary improvement is below the frozen `epsilon=0.002`, so it is not
reported as significant. The organizer's hidden baseline is primary `0.5946`;
our hidden score and hidden delta are pending organizer evaluation and are not
estimated here.

KuaiRand-1K and KuaiRand-27K were not attempted.

## Judged-run resources

| Resource | Reported value |
|---|---:|
| Iterations | 9 / 50 |
| Agent wall-clock to recorded convergence stop | 1,437.671 seconds / 0.3994 hours |
| Training-through-trial-09 completion | 1,312.570 seconds / 0.3646 hours |
| LLM input tokens through convergence | 69,244,629 |
| Cached input tokens included above | 67,277,696 |
| LLM output tokens through convergence | 365,101 |
| Total input + output tokens | 69,609,730 |
| In-run coding-provider calls | 0 |
| GPU usage | 0 seconds / 0 GPU-hours |
| Manual research interventions | 1 |
| Test labels loaded | false |
| Test metrics used | false |

Token counts come from the Codex host's cumulative telemetry for this Agent AI-yoh
task at `2026-08-31T19:02:50.527Z`, the final usage event before the recorded
convergence stop. They include the full development/research task through
convergence and are deliberately more conservative than the judged harness's
in-loop provider counter. Reasoning tokens are included within the reported
output accounting supplied by the host.

## Final artifacts

| Artifact | Value |
|---|---|
| Locked trial | trial-07 |
| Checkpoint | `ranker.npz`, 3.3 MB |
| Checkpoint SHA-256 | `e703d6a26a82b53bb14bf7dca3b8c736de1d6fb8328ff538fe529635286752da` |
| Score file | `submission_trial07.csv`, 170,588 rows, 4.3 MB |
| Score-file SHA-256 | `8237ab9da63215ec68a2801f0aa0aa717a8acab26f7d9dd8012838c8a18b032a` |
| Submission schema | `row_id,user_id,video_id,score` |
| Schema/alignment/finite-score check | passed |
| Decision journal | 21 append-only, hash-chained decisions |

The score file was generated from the immutable validation lock. No hidden
metric was computed. Do not run `submit.py --score --split test`.
