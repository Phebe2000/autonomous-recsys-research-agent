"""Three validation-only fixed-negative-gate controlled runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from data import encode, load

from .fingerprint import canonical_json
from .history import build_causal_history
from .history_pipeline import _load_baseline, _train_history
from .listwise import group_user_exposures
from .onboarding import onboard
from .runner import TrialRunner, TrialSpec
from .selection import write_top1
from .store import ResearchStore


def _atomic_json(path: Path, payload: dict) -> None:
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


def _config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def _checkpoint_scalars(trial: dict) -> dict | None:
    if trial["status"] != "completed":
        return None
    path = Path(trial["result"]["artifacts"][0]["path"])
    with np.load(path, allow_pickle=False) as checkpoint:
        return {
            "best_gate": float(checkpoint["gate"]),
            "optimizer_steps": int(checkpoint["t"]),
            "history_optimizer_steps": int(checkpoint["tH"]),
        }


def run_controlled(data_dir: Path, state_root: Path) -> dict:
    onboarding = onboard(data_dir, state_root, purpose="development")
    fingerprint = onboarding["manifest"]["dataset_fingerprint"]
    state_dir = Path(onboarding["state_dir"])
    store = ResearchStore(Path(onboarding["ledger"]), fingerprint)
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    baseline_model, _, baseline_artifact = _load_baseline(store, dim, fingerprint)
    config_dir = Path(__file__).with_name("configs")
    prior = json.loads((config_dir / "listnet_prior.json").read_text())
    controlled = json.loads((config_dir / "history_fixed_gate_controlled.json").read_text())

    diagnostic_path = state_dir / "history_gate_diagnostic_results.json"
    if not diagnostic_path.exists():
        raise RuntimeError("fixed-gate runs require the completed G-01 diagnostic")
    diagnostic = json.loads(diagnostic_path.read_text())
    reference = store.get(diagnostic["runs"]["G-01"]["trial_id"])
    if reference["status"] != "completed":
        raise RuntimeError("F-00 reference G-01 must be completed")

    history_index = build_causal_history(data_dir, splits, enc)
    train_scores = baseline_model.predict(enc["train"][0])
    groups = group_user_exposures(
        enc["train"][2], enc["train"][1], train_scores,
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    run_trials = {}
    for run_id, changes in controlled["runs"].items():
        history_config = {**controlled["fixed"], **changes}
        config = {
            "schema_version": 1,
            "run_id": run_id,
            "method": controlled["method"],
            "seed": 0,
            "listwise_prior": prior,
            "history": history_config,
            "initialization": {
                "source": "fresh_validation_best_official_fm_on_current_dataset",
                "baseline_artifact_sha256": baseline_artifact["sha256"],
            },
        }
        matches = [
            trial for trial in store.trials()
            if trial["method"] == controlled["method"]
            and trial["config_hash"] == _config_hash(config)
        ]
        if matches and matches[0]["status"] in {"completed", "failed", "pruned"}:
            run_trials[run_id] = matches[0]
            continue
        resume_id = matches[0]["trial_id"] if matches else None
        spec = TrialSpec(controlled["method"], changes["description"], config, 0)

        def candidate(trial, current_config=config):
            return _train_history(
                trial, enc, baseline_model, groups, history_index,
                current_config, state_dir, fingerprint,
            )

        run_trials[run_id] = runner.execute(spec, candidate, resume_trial_id=resume_id)

    reference_primary = reference["validation"]["primary"]
    results = {
        "F-00": {
            "source": "G-01 learned-gate reference",
            "source_trial_id": reference["trial_id"],
            "new_budget_cost": 0,
            "status": reference["status"],
            "validation": reference["validation"],
            "gain_vs_F-00": 0.0,
            **(_checkpoint_scalars(reference) or {}),
        }
    }
    for run_id, trial in run_trials.items():
        validation = trial.get("validation")
        results[run_id] = {
            "trial_id": trial["trial_id"],
            "status": trial["status"],
            "validation": validation,
            "gain_vs_F-00": None if validation is None else validation["primary"] - reference_primary,
            **(_checkpoint_scalars(trial) or {}),
        }
    top1 = write_top1(store, state_dir / "top1.json")
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "objective": "pure Soft-target ListNet; no BPR",
        "selection": "validation primary only",
        "test_metrics_used": False,
        "runs": results,
        "validation_top1": top1,
        "ledger": {"used": store.consumed, "remaining": store.remaining},
    }
    _atomic_json(state_dir / "history_fixed_gate_results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="KuaiRand-Pure/data")
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    args = parser.parse_args()
    started = time.time()
    result = run_controlled(Path(args.data_dir), Path(args.state_root))
    result["wall_time_sec"] = time.time() - started
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
