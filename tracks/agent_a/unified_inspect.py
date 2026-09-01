"""Inspect unified candidates and evidence without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import anchor_candidates, build_evidence_registry, write_evidence_reports
from .fingerprint import fingerprint_dataset
from .store import ResearchStore
from .unified_runner import UnifiedTrialRunner


def inspect(data_dir: Path, state_root: Path, write_reports: bool = True) -> dict:
    identity = fingerprint_dataset(data_dir)
    fingerprint = identity["dataset_fingerprint"]
    state_dir = Path(state_root) / fingerprint.removeprefix("sha256:")
    ledger_path = state_dir / "research.sqlite3"
    if not ledger_path.exists():
        raise FileNotFoundError("dataset must be onboarded before unified inspection")
    store = ResearchStore(ledger_path, fingerprint)
    before = store.consumed
    registry = build_evidence_registry(store)
    facade = UnifiedTrialRunner(store)
    candidates = {
        name: {
            "spec": spec.to_dict(),
            "dispatch": facade.inspect(spec),
            "evidence": registry["modules"].get(name),
        }
        for name, spec in anchor_candidates(fingerprint).items()
    }
    reports = None
    if write_reports:
        json_path, markdown_path = write_evidence_reports(registry, state_dir)
        reports = {"json": str(json_path), "markdown": str(markdown_path)}
    after = store.consumed
    if before != after:
        raise RuntimeError("inspect mutated the dataset trial ledger")
    return {
        "mode": "inspect_only_no_training",
        "dataset_fingerprint": fingerprint,
        "budget_before": {"used": before, "remaining": store.max_trials - before},
        "budget_after": {"used": after, "remaining": store.max_trials - after},
        "top1": registry["top1"],
        "candidates": candidates,
        "evidence_summary": {
            name: {
                "classification": module["classification"],
                "observed_on_current_fingerprint": module["observed_on_current_fingerprint"],
                "eligible_as_new_dataset_anchor": module["eligible_as_new_dataset_anchor"],
                "enabled_for_current_search": module["enabled_for_current_search"],
            }
            for name, module in registry["modules"].items()
        },
        "next_step": {
            "current_fingerprint": registry["default_current_candidate_modules"],
            "new_fingerprint": "run each eligible anchor once under its independent ledger before enabling search",
        },
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="KuaiRand-Pure/data")
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    parser.add_argument("--no-write-reports", action="store_true")
    args = parser.parse_args()
    result = inspect(Path(args.data_dir), Path(args.state_root), not args.no_write_reports)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
