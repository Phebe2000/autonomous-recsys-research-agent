"""Two-trial gate=0 and delayed-history-unfreeze diagnostic."""

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
    prior = json.loads(
        (Path(__file__).with_name("configs") / "listnet_prior.json").read_text()
    )
    diagnostic = json.loads(
        (Path(__file__).with_name("configs") / "history_gate_diagnostic.json").read_text()
    )
    h00 = store.get("trial-04")
    if h00["status"] != "completed" or h00["method"] != prior["method"]:
        raise RuntimeError("G-00 requires reusable no-history trial-04")
    history_index = build_causal_history(data_dir, splits, enc)
    train_scores = baseline_model.predict(enc["train"][0])
    groups = group_user_exposures(
        enc["train"][2],
        enc["train"][1],
        train_scores,
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    run_trials = {}
    for run_id, changes in diagnostic["runs"].items():
        history_config = {**diagnostic["fixed"], **changes}
        config = {
            "schema_version": 1,
            "run_id": run_id,
            "method": diagnostic["method"],
            "seed": 0,
            "listwise_prior": prior,
            "history": history_config,
            "initialization": {
                "source": "fresh_validation_best_official_fm_on_current_dataset",
                "baseline_artifact_sha256": baseline_artifact["sha256"],
            },
        }
        matches = [
            trial
            for trial in store.trials()
            if trial["method"] == diagnostic["method"]
            and trial["config_hash"] == _config_hash(config)
        ]
        if matches and matches[0]["status"] in {"completed", "failed", "pruned"}:
            run_trials[run_id] = matches[0]
            continue
        resume_id = matches[0]["trial_id"] if matches else None
        spec = TrialSpec(
            diagnostic["method"],
            changes["description"],
            config,
            0,
        )

        def candidate(trial, current_config=config):
            return _train_history(
                trial,
                enc,
                baseline_model,
                groups,
                history_index,
                current_config,
                state_dir,
                fingerprint,
            )

        run_trials[run_id] = runner.execute(spec, candidate, resume_trial_id=resume_id)

    base_primary = h00["validation"]["primary"]
    results = {
        "G-00": {
            "source_trial_id": h00["trial_id"],
            "new_budget_cost": 0,
            "status": h00["status"],
            "validation": h00["validation"],
            "gain_vs_G-00": 0.0,
        }
    }
    for run_id, trial in run_trials.items():
        validation = trial.get("validation")
        results[run_id] = {
            "trial_id": trial["trial_id"],
            "status": trial["status"],
            "validation": validation,
            "gain_vs_G-00": None if validation is None else validation["primary"] - base_primary,
            **(_checkpoint_scalars(trial) or {}),
        }
    write_top1(store, state_dir / "top1.json")
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "objective": "pure Soft-target ListNet; no BPR",
        "selection": "validation primary only",
        "test_metrics_used": False,
        "runs": results,
        "ledger": {"used": store.consumed, "remaining": store.remaining},
    }
    _atomic_json(state_dir / "history_gate_diagnostic_results.json", payload)
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
