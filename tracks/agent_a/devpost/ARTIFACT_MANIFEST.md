# Final artifact manifest

This manifest identifies the immutable validation lock, final label-free score
file, and audit records packaged for Devpost. All hashes use SHA-256.

| Artifact | SHA-256 |
|---|---|
| `submission_trial07.csv` | `8237ab9da63215ec68a2801f0aa0aa717a8acab26f7d9dd8012838c8a18b032a` |
| `submission_trial07.csv.manifest.json` | `490ea58df0bbc1752d2338a5f3063460c94016efe45642e8ac6212b5482ed36e` |
| `ranker.npz` | `e703d6a26a82b53bb14bf7dca3b8c736de1d6fb8328ff538fe529635286752da` |
| `agent_decision_journal.json` | `98cad0dd97085717d06d554f591778f7f9856e3a1d3948e867ee5202255bc104` |
| `iteration_audit_log.json` | `fe1fedfd71a5e830098598196dbb6a6221b05af51d757a4115c8df1fa7a01924` |
| `locked_manifest.json` | `bee9c57a7cf829f2edc6f9ba2ce794815503c51a14b472df82cbd2278ecf1a83` |
| `resource_usage.json` | `0792a2c4bee0dfc3cd27398cc59b5e733868948a1af273ddc6e91a78068d3658` |
| `run_policy.json` | `00e5859155f7ab403fb2723083405a0222a45c0b04056ae1f6fe2777e5ba4cc9` |
| `run_timing.json` | `50c6838185407a080c90a75bfe64d665336d1957878707c5c65ca4604f05db6e` |
| `top1.json` | `ae612faab6ef5a3e62311d4409c9e41f3021d3e45e52e968de5571fa010137c6` |
| `research.sqlite3` | `391d3d8a5a043bc6267a5178989ab66c837be3ce4bb1e388eeff51cfcd4d86b9` |
| `optuna.sqlite3` | `4c43aca467daf90d5a81af00a8b8de2971587022f59cec94c33d49b686156d7b` |

The package intentionally contains no raw KuaiRand data, credentials, virtual
environment, test label, or hidden-test metric. The score file is the artifact
to submit for organizer scoring; `ranker.npz` is the locked model checkpoint.
