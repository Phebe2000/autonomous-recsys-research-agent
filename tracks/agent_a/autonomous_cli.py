"""CLI for planning, simulating, running, resuming, and locking research loops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from .autonomous import AutonomousResearchLoop, SEARCH_SCHEMA, SyntheticExecutor, phase_for_ordinal
from .evidence import AUDITED_KUAIRAND_FINGERPRINT
from .fingerprint import fingerprint_dataset
from .real_executor import RealCandidateExecutor


def _state_dir(state_root: Path, fingerprint: str) -> Path:
    return Path(state_root) / fingerprint.removeprefix("sha256:")


def _read_budget(state_dir: Path) -> tuple[int, int]:
    ledger = state_dir / "research.sqlite3"
    if not ledger.exists():
        return 0, 50
    connection = sqlite3.connect(ledger)
    try:
        used = int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    finally:
        connection.close()
    return used, 50 - used


def plan(data_dir: Path, state_root: Path) -> dict:
    fingerprint = fingerprint_dataset(data_dir)["dataset_fingerprint"]
    state_dir = _state_dir(state_root, fingerprint)
    used, remaining = _read_budget(state_dir)
    evidence = None
    evidence_path = state_dir / "evidence_registry.json"
    if evidence_path.exists():
        registry = json.loads(evidence_path.read_text())
        if registry.get("dataset_fingerprint") == fingerprint:
            evidence = {
                "default_current_candidate_modules": registry.get("default_current_candidate_modules", []),
                "modules": {
                    name: {
                        "classification": module["classification"],
                        "enabled_for_current_search": module["enabled_for_current_search"],
                        "eligible_as_new_dataset_anchor": module["eligible_as_new_dataset_anchor"],
                    }
                    for name, module in registry.get("modules", {}).items()
                },
            }
    return {
        "command": "plan",
        "read_only": True,
        "dataset_fingerprint": fingerprint,
        "budget": {"used": used, "remaining": remaining, "maximum": 50},
        "phase": phase_for_ordinal(min(used + 1, 50)),
        "trial_mapping": [],
        "best_validation": None,
        "stop_reason": "budget_exhausted" if used >= 50 else None,
        "search_space": json.loads(SEARCH_SCHEMA.read_text()),
        "evidence": evidence,
        "test_metrics_used": False,
    }


def inspect(data_dir: Path, state_root: Path) -> dict:
    payload = plan(data_dir, state_root)
    payload["command"] = "inspect"
    fingerprint = payload["dataset_fingerprint"]
    state_dir = _state_dir(state_root, fingerprint)
    ledger = state_dir / "research.sqlite3"
    if ledger.exists():
        connection = sqlite3.connect(ledger)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT trial_id,status,validation_json,config_json FROM trials ORDER BY ordinal"
            ).fetchall()
        finally:
            connection.close()
        payload["trial_mapping"] = [
            {
                "ledger_trial_id": row["trial_id"],
                "status": row["status"],
                "optuna_trial_number": json.loads(row["config_json"]).get("optuna", {}).get("trial_number"),
            }
            for row in rows
        ]
        completed = [row for row in rows if row["validation_json"]]
        if completed:
            best = max(
                completed,
                key=lambda row: json.loads(row["validation_json"])["primary"],
            )
            payload["best_validation"] = json.loads(best["validation_json"])
    return payload


def simulate(
    state_root: Path | None,
    max_trials: int,
    target_trials: int | None = None,
) -> dict:
    root = Path(tempfile.mkdtemp(prefix="agent-a-simulation-")) if state_root is None else Path(state_root)
    production_runtime = Path(__file__).with_name("runtime").resolve()
    resolved_root = root.resolve()
    if resolved_root == production_runtime or production_runtime in resolved_root.parents:
        raise ValueError("simulation state must not be placed in the production runtime tree")
    fingerprint = "sha256:synthetic-agent-a-tpe-v1"
    state_dir = _state_dir(root, fingerprint)
    loop = AutonomousResearchLoop(
        state_dir,
        fingerprint,
        simulation=True,
        max_trials=max_trials,
        convergence_enabled=False,
    )
    result = loop.run(
        SyntheticExecutor(),
        target_trials=max_trials if target_trials is None else target_trials,
    )
    result["command"] = "simulate"
    result["state_root"] = str(root)
    result["synthetic_fingerprint"] = fingerprint
    return result


def run_real(data_dir: Path, state_root: Path, resume: bool) -> dict:
    fingerprint = fingerprint_dataset(data_dir)["dataset_fingerprint"]
    if fingerprint == AUDITED_KUAIRAND_FINGERPRINT:
        raise RuntimeError("new training on the audited KuaiRand fingerprint is forbidden")
    state_dir = _state_dir(state_root, fingerprint)
    loop = AutonomousResearchLoop(state_dir, fingerprint, simulation=False, max_trials=50)
    executor = RealCandidateExecutor(data_dir, state_dir, loop.store, fingerprint)
    result = loop.run(executor)
    result["command"] = "resume" if resume else "run"
    return result


def lock(data_dir: Path, state_root: Path) -> dict:
    fingerprint = fingerprint_dataset(data_dir)["dataset_fingerprint"]
    state_dir = _state_dir(state_root, fingerprint)
    if not (state_dir / "research.sqlite3").exists():
        raise FileNotFoundError("cannot lock a dataset without a research ledger")
    loop = AutonomousResearchLoop(
        state_dir,
        fingerprint,
        simulation=fingerprint.startswith("sha256:synthetic-"),
        max_trials=50,
    )
    manifest = loop.lock()
    report = loop.report()
    return {"command": "lock", "manifest": manifest, **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "inspect", "run", "resume", "lock"):
        command = subparsers.add_parser(name)
        command.add_argument("--data-dir", default="KuaiRand-Pure/data")
        command.add_argument("--state-root", default="tracks/agent_a/runtime")
    simulation = subparsers.add_parser("simulate")
    simulation.add_argument("--state-root")
    simulation.add_argument("--max-trials", type=int, default=50)
    simulation.add_argument("--target-trials", type=int)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(Path(args.data_dir), Path(args.state_root))
    elif args.command == "inspect":
        result = inspect(Path(args.data_dir), Path(args.state_root))
    elif args.command == "simulate":
        result = simulate(
            None if args.state_root is None else Path(args.state_root),
            args.max_trials,
            args.target_trials,
        )
    elif args.command in {"run", "resume"}:
        result = run_real(Path(args.data_dir), Path(args.state_root), args.command == "resume")
    else:
        result = lock(Path(args.data_dir), Path(args.state_root))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
