"""Paired cross-seed replication for the fixed negative history gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

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


def run_replication(data_dir: Path, state_root: Path) -> dict:
    onboarding = onboard(data_dir, state_root, purpose="development")
    fingerprint = onboarding["manifest"]["dataset_fingerprint"]
    state_dir = Path(onboarding["state_dir"])
    store = ResearchStore(Path(onboarding["ledger"]), fingerprint)
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    baseline_model, _, baseline_artifact = _load_baseline(store, dim, fingerprint)
    config_dir = Path(__file__).with_name("configs")
    prior = json.loads((config_dir / "listnet_prior.json").read_text())
    replication = json.loads((config_dir / "history_seed_replication.json").read_text())
    history_index = build_causal_history(data_dir, splits, enc)
    groups = group_user_exposures(
        enc["train"][2], enc["train"][1], baseline_model.predict(enc["train"][0]),
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    trials = {}
    for seed in replication["seeds"]:
        for variant, changes in replication["variants"].items():
            run_id = f"S{seed}-{variant}"
            seeded_prior = {**prior, "seed": seed}
            history_config = {**replication["fixed"], **changes}
            config = {
                "schema_version": 1,
                "run_id": run_id,
                "method": replication["method"],
                "seed": seed,
                "listwise_prior": seeded_prior,
                "history": history_config,
                "paired_design": {
                    "pair_seed": seed,
                    "variant": variant,
                    "only_difference_within_pair": "history_gate_initial",
                },
                "initialization": {
                    "source": "same_fresh_validation_best_seed0_official_fm_for_all_pairs",
                    "baseline_artifact_sha256": baseline_artifact["sha256"],
                },
            }
            matches = [
                trial for trial in store.trials()
                if trial["method"] == replication["method"]
                and trial["config_hash"] == _config_hash(config)
            ]
            if matches and matches[0]["status"] in {"completed", "failed", "pruned"}:
                trials[run_id] = matches[0]
                continue
            resume_id = matches[0]["trial_id"] if matches else None
            spec = TrialSpec(replication["method"], changes["description"], config, seed)

            def candidate(trial, current_config=config):
                return _train_history(
                    trial, enc, baseline_model, groups, history_index,
                    current_config, state_dir, fingerprint,
                )

            trials[run_id] = runner.execute(spec, candidate, resume_trial_id=resume_id)

    pairs = {}
    for seed in replication["seeds"]:
        control = trials[f"S{seed}-control"]
        history = trials[f"S{seed}-history"]
        control_validation = control.get("validation")
        history_validation = history.get("validation")
        gain = None
        if control_validation is not None and history_validation is not None:
            gain = history_validation["primary"] - control_validation["primary"]
        pairs[str(seed)] = {
            "control": {
                "trial_id": control["trial_id"],
                "status": control["status"],
                "validation": control_validation,
            },
            "history": {
                "trial_id": history["trial_id"],
                "status": history["status"],
                "validation": history_validation,
            },
            "history_primary_gain": gain,
        }
    top1 = write_top1(store, state_dir / "top1.json")
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "design": "paired training-shuffle seeds; shared baseline initialization",
        "objective": "pure Soft-target ListNet; no BPR",
        "selection": "validation primary only",
        "test_metrics_used": False,
        "pairs": pairs,
        "validation_top1": top1,
        "ledger": {"used": store.consumed, "remaining": store.remaining},
    }
    _atomic_json(state_dir / "history_seed_replication_results.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="KuaiRand-Pure/data")
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    args = parser.parse_args()
    started = time.time()
    result = run_replication(Path(args.data_dir), Path(args.state_root))
    result["wall_time_sec"] = time.time() - started
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
