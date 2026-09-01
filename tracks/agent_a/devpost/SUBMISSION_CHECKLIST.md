# Devpost submission checklist

## Written description

- [x] Problem statement and solution explained
- [x] Development tools listed
- [x] APIs/model services listed
- [x] Libraries and versions listed
- [x] Datasets and assets listed
- [x] Team name and all four team members recorded
- [ ] Paste `DEVPOST_PROJECT_DESCRIPTION.md` into Devpost and review formatting

## Public repository

- [ ] Confirm <https://github.com/Phebe2000/autonomous-recsys-research-agent> is Public
- [ ] Merge or make `codex/agent-a` the judge-visible default branch
- [x] Root README contains overview, setup, reproduction, limitations, and contributions
- [x] Source code and 82-test compliance suite are documented
- [x] Raw dataset, virtual environment, credentials, and SQLite stores remain excluded

## Run and iteration logs

- [x] Nine hypotheses documented
- [x] Code-diff provenance and SHA-256 values documented
- [x] GAUC and nDCG@5 documented for every trial
- [x] Errors/recovery behavior documented
- [x] Manual-intervention count reported as 1
- [x] Machine-readable audit and decision journal included in local artifact package

## Final result

- [x] Immutable validation lock points to trial-07
- [x] Checkpoint SHA-256 verified
- [x] `submission_trial07.csv` has 170,588 aligned finite scores
- [x] Submission SHA-256 verified
- [x] Validation result and approximate official-baseline deltas reported
- [x] Iterations, wall-clock, tokens, GPU-hours, and interventions reported
- [x] Bonus benchmarks clearly reported as not attempted
- [ ] Upload `agent_a_devpost_submission.zip` or its required files to Devpost
- [ ] Submit the score file once for organizer hidden evaluation
- [ ] Record organizer hidden result without restarting HPO or changing the lock
