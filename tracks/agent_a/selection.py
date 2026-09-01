"""Deterministic Top-1 materialization from validation-only ledger fields."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .contracts import reject_test_data
from .store import ResearchStore


def write_top1(store: ResearchStore, path: Path) -> dict:
    trial = store.best_trial()
    if trial is None:
        raise RuntimeError("no completed validation trial is available")
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": store.dataset_fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "split": "valid",
            "metric": "primary",
            "direction": "max",
            "tie_break": "lowest_trial_ordinal",
        },
        "trial": {
            "trial_id": trial["trial_id"],
            "ordinal": trial["ordinal"],
            "method": trial["method"],
            "seed": trial["seed"],
            "config_hash": trial["config_hash"],
            "config": trial["config"],
        },
        "validation": trial["validation"],
        "artifacts": (trial["result"] or {}).get("artifacts", []),
        "test_metrics_used": False,
        "source_ledger_sha256": hashlib.sha256(store.path.read_bytes()).hexdigest(),
    }
    reject_test_data({key: value for key, value in payload.items() if key != "test_metrics_used"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload
