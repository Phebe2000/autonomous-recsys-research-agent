"""Compliance-first CLI for the required KuaiRand-Pure judged run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .autonomous import AutonomousResearchLoop, phase_for_ordinal
from .compliance import JudgedRunCompliance, JudgedRunPolicy, _atomic_json
from .codegen import CodeGeneratingResearchLoop, CodexCodingProvider
from .fingerprint import fingerprint_judged_dataset
from .real_executor import RealCandidateExecutor
from .store import ResearchStore


DEFAULT_STATE_ROOT = Path("tracks/agent_a/judged_runtime")


def _state(data_dir: Path, state_root: Path) -> tuple[dict, Path]:
    manifest = fingerprint_judged_dataset(data_dir)
    fingerprint = manifest["dataset_fingerprint"]
    return manifest, Path(state_root) / fingerprint.removeprefix("sha256:")


def _code_snapshot(repo_root: Path) -> dict:
    files = [repo_root / name for name in ("data.py", "baseline.py", "evaluate.py", "submit.py")]
    files += sorted((repo_root / "tracks" / "agent_a").glob("*.py"))
    files += sorted((repo_root / "tracks" / "agent_a" / "configs").glob("*.json"))
    files += sorted((repo_root / "tracks" / "agent_a").glob("requirements*.txt"))
    records = []
    for path in files:
        relative = str(path.relative_to(repo_root))
        records.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "aggregate_sha256": digest, "files": records}


def plan(data_dir: Path, state_root: Path) -> dict:
    manifest, state_dir = _state(data_dir, state_root)
    ledger = state_dir / "research.sqlite3"
    used = 0
    if ledger.exists():
        connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
        try:
            used = int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
        finally:
            connection.close()
    policy = None
    if (state_dir / "run_policy.json").exists():
        policy = json.loads((state_dir / "run_policy.json").read_text())
    return {
        "command": "plan",
        "read_only": True,
        "benchmark": "KuaiRand-Pure",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "fingerprint_algorithm": manifest["fingerprint_algorithm"],
        "hidden_test_labels_in_fingerprint": False,
        "random_log_used_for_training": False,
        "budget": {"used": used, "remaining": 50 - used, "maximum": 50},
        "phase": phase_for_ordinal(min(used + 1, 50)),
        "initialized": ledger.exists() and policy is not None,
        "policy": policy,
        "test_metrics_used": False,
        "hidden_labels_loaded": False,
    }


def initialize(
    data_dir: Path,
    state_root: Path,
    *,
    epsilon: float = 0.002,
    convergence_n: int = 3,
    minimum_scored_iterations: int = 9,
) -> dict:
    manifest, state_dir = _state(data_dir, state_root)
    fingerprint = manifest["dataset_fingerprint"]
    policy = JudgedRunPolicy(
        fingerprint,
        epsilon=epsilon,
        convergence_n=convergence_n,
        minimum_scored_iterations=minimum_scored_iterations,
    )
    store = ResearchStore(state_dir / "research.sqlite3", fingerprint, max_trials=50)
    if store.consumed:
        raise RuntimeError("judged-run initialization requires an existing 0/50 ledger")
    compliance = JudgedRunCompliance(state_dir, policy)
    repo_root = Path(__file__).resolve().parents[2]
    snapshot_path = state_dir / "initial_code_snapshot.json"
    snapshot = _code_snapshot(repo_root)
    if snapshot_path.exists() and json.loads(snapshot_path.read_text()) != snapshot:
        raise ValueError("initial judged-run code snapshot changed before iteration 1")
    if not snapshot_path.exists():
        _atomic_json(snapshot_path, snapshot)
    audit = compliance.write_audit(store)
    _atomic_json(state_dir / "dataset_manifest.json", manifest)
    return {
        "command": "init",
        "state_dir": str(state_dir),
        "dataset_fingerprint": fingerprint,
        "policy": policy.to_dict(),
        "initial_code_sha256": snapshot["aggregate_sha256"],
        "budget": audit["budget"],
        "clock_started": False,
        "test_metrics_used": False,
        "hidden_labels_loaded": False,
    }


def supersede_empty_preflight(data_dir: Path, state_root: Path, reason: str) -> dict:
    """Archive an unstarted 0/50 preflight after the agent code changes."""
    if not reason.strip():
        raise ValueError("supersession requires a non-empty audit reason")
    _, state_dir = _state(data_dir, state_root)
    ledger = state_dir / "research.sqlite3"
    timing_path = state_dir / "run_timing.json"
    policy_path = state_dir / "run_policy.json"
    if not (ledger.exists() and timing_path.exists() and policy_path.exists()):
        raise FileNotFoundError("no initialized preflight exists to supersede")
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        used = int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    finally:
        connection.close()
    timing = json.loads(timing_path.read_text())
    if used != 0 or timing["started_at"] is not None:
        raise RuntimeError("only an unstarted 0/50 preflight may be superseded")
    policy = json.loads(policy_path.read_text())
    snapshot = json.loads((state_dir / "initial_code_snapshot.json").read_text())
    archive_name = f"{policy['policy_identity']}-{snapshot['aggregate_sha256'][:12]}"
    archive = state_dir / "superseded" / archive_name
    if archive.exists():
        raise FileExistsError("this preflight policy was already archived")
    archive.mkdir(parents=True)
    for path in list(state_dir.iterdir()):
        if path.name != "superseded":
            path.replace(archive / path.name)
    marker = {
        "schema_version": 1,
        "superseded_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "budget_used": 0,
        "clock_started": False,
        "prior_policy_identity": policy["policy_identity"],
        "prior_initial_code_sha256": snapshot["aggregate_sha256"],
        "recoverable": True,
    }
    _atomic_json(archive / "superseded.json", marker)
    return {"command": "supersede-empty", "archive": str(archive), **marker}


def _active(data_dir: Path, state_root: Path):
    manifest, state_dir = _state(data_dir, state_root)
    if not (state_dir / "run_policy.json").exists():
        raise FileNotFoundError("initialize the judged run before run/resume/lock")
    policy_data = json.loads((state_dir / "run_policy.json").read_text())
    policy_data.pop("policy_identity", None)
    policy = JudgedRunPolicy(**policy_data)
    compliance = JudgedRunCompliance(state_dir, policy)
    loop = AutonomousResearchLoop(
        state_dir,
        manifest["dataset_fingerprint"],
        simulation=False,
        max_trials=policy.max_iterations,
        convergence_enabled=True,
        compliance=compliance,
    )
    return state_dir, compliance, loop


def run(
    data_dir: Path,
    state_root: Path,
    resume: bool,
    *,
    provider_model: str | None = None,
    target_trials: int | None = None,
) -> dict:
    state_dir, compliance, loop = _active(data_dir, state_root)
    real_executor = RealCandidateExecutor(data_dir, state_dir, loop.store, loop.fingerprint)

    def executor(trial, spec, module, reporter):
        return compliance.execute_with_deadline(
            lambda: real_executor(trial, spec, module, reporter)
        )

    target = loop.max_trials if target_trials is None else min(target_trials, loop.max_trials)
    events = []
    if loop.store.consumed < min(6, target) and loop.stop_reason() is None:
        anchor_report = loop.run(executor, target_trials=min(6, target))
        events.extend(anchor_report["events"])
    if loop.store.consumed < target and loop.stop_reason() is None:
        provider = CodexCodingProvider(
            Path(__file__).resolve().parents[2], model=provider_model
        )
        codegen = CodeGeneratingResearchLoop(loop, compliance, real_executor, provider)
        while loop.store.consumed < target and loop.stop_reason() is None:
            ordinal = loop.store.consumed + 1
            # Preserve both auditable code-generation autonomy and conditional
            # Optuna refinement of positive preimplemented modules. Odd trials
            # use the unified TPE/config runner; even trials use generated code.
            if ordinal % 2:
                events.append(loop.step(executor))
            else:
                events.append(codegen.step())
        result = loop.report(events)
        result["research_mode"] = "hybrid_optuna_and_llm_code_generation"
    else:
        result = loop.report(events)
        result["research_mode"] = "controlled_anchors_then_llm_code_generation"
    result["command"] = "resume" if resume else "run"
    result["audit_log"] = str(compliance.audit_path)
    return result


def inspect(data_dir: Path, state_root: Path) -> dict:
    payload = plan(data_dir, state_root)
    payload["command"] = "inspect"
    _, state_dir = _state(data_dir, state_root)
    if (state_dir / "iteration_audit_log.json").exists():
        audit = json.loads((state_dir / "iteration_audit_log.json").read_text())
        payload["timing"] = audit["timing"]
        payload["resource_usage"] = audit["resource_usage"]
        payload["iterations"] = audit["iterations"]
    return payload


def lock(data_dir: Path, state_root: Path) -> dict:
    _, compliance, loop = _active(data_dir, state_root)
    manifest = loop.lock()
    return {"command": "lock", "manifest": manifest, "report": loop.report(), "audit": compliance.write_audit(loop.store)}


def usage(data_dir: Path, state_root: Path, args) -> dict:
    _, compliance, loop = _active(data_dir, state_root)
    totals = compliance.record_usage(
        llm_input_tokens=args.llm_input_tokens,
        llm_output_tokens=args.llm_output_tokens,
        gpu_seconds=args.gpu_seconds,
        manual_interventions=args.manual_interventions,
    )
    compliance.write_audit(loop.store)
    return {"command": "record-usage", "resource_usage": totals}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "init", "inspect", "run", "resume", "lock", "supersede-empty"):
        command = commands.add_parser(name)
        command.add_argument("--data-dir", default="KuaiRand-Pure/data")
        command.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
        if name == "init":
            command.add_argument("--epsilon", type=float, default=0.002)
            command.add_argument("--convergence-n", type=int, default=3)
            command.add_argument("--minimum-scored-iterations", type=int, default=9)
        if name in {"run", "resume"}:
            command.add_argument("--provider-model", default="gpt-5.6-sol")
            command.add_argument("--target-trials", type=int)
        if name == "supersede-empty":
            command.add_argument("--reason", required=True)
    record = commands.add_parser("record-usage")
    record.add_argument("--data-dir", default="KuaiRand-Pure/data")
    record.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    record.add_argument("--llm-input-tokens", type=int, default=0)
    record.add_argument("--llm-output-tokens", type=int, default=0)
    record.add_argument("--gpu-seconds", type=float, default=0.0)
    record.add_argument("--manual-interventions", type=int, default=0)
    args = parser.parse_args()
    data_dir, state_root = Path(args.data_dir), Path(args.state_root)
    if args.command == "plan":
        result = plan(data_dir, state_root)
    elif args.command == "init":
        result = initialize(
            data_dir,
            state_root,
            epsilon=args.epsilon,
            convergence_n=args.convergence_n,
            minimum_scored_iterations=args.minimum_scored_iterations,
        )
    elif args.command == "inspect":
        result = inspect(data_dir, state_root)
    elif args.command == "supersede-empty":
        result = supersede_empty_preflight(data_dir, state_root, args.reason)
    elif args.command in {"run", "resume"}:
        result = run(
            data_dir,
            state_root,
            args.command == "resume",
            provider_model=args.provider_model,
            target_trials=args.target_trials,
        )
    elif args.command == "lock":
        result = lock(data_dir, state_root)
    else:
        result = usage(data_dir, state_root, args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
