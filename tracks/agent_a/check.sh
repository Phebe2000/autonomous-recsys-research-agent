#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

.venv/bin/python -W error -m unittest discover -s tracks/agent_a/tests -v
.venv/bin/python -m compileall -q tracks/agent_a
git diff --check
git diff --exit-code -- data.py baseline.py evaluate.py submit.py
.venv/bin/python -m tracks.agent_a.readiness_audit
