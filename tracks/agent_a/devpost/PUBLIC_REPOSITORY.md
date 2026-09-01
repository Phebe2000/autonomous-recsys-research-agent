# Public code repository

Repository: <https://github.com/Phebe2000/autonomous-recsys-research-agent>

Submission branch: `codex/agent-a`

The repository contains the unchanged organizer Starter Kit and the Agent AI-yoh
implementation under `tracks/agent_a/`.

## Component map

| Area | Files |
|---|---|
| Candidate schema and identity | `candidate.py`, `unified_runner.py` |
| Safe data and fingerprinting | `safe_data.py`, `fingerprint.py` |
| Candidate models | `listwise.py`, `history.py`, `bpr.py`, `multitask.py`, `ranker.py` |
| Causal features | `behavioral_features.py`, `history_data.py` |
| Training execution | `real_executor.py`, `runner.py` |
| Autonomous control | `autonomous.py`, `codegen.py`, `store.py` |
| Compliance and CLI | `compliance.py`, `judged_cli.py`, `finalize.py` |
| Submission safety | `safe_submit_check.py`, official `submit.py` writer |
| Tests | `tracks/agent_a/tests/` |
| Documentation | `README.md`, workflow documents, runbook, `devpost/` |

## Public-release checks

- Confirm GitHub repository visibility is set to **Public**.
- Merge or expose `codex/agent-a` from the repository's default branch so judges
  do not need a private branch reference.
- Keep KuaiRand raw data, `.venv`, SQLite runtime stores, and local credentials
  out of Git.
- Include the English README, setup commands, reproduction procedure,
  limitations, team roster, and contribution statement.
- Attach the separate final artifact package through Devpost if binary outputs
  are not stored in Git.

At preparation time, the Git remote points to the URL above. Repository public
visibility must still be confirmed in GitHub settings before final submission.
