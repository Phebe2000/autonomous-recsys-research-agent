"""Controlled validation-only BPR-weight runs on the no-history ListNet model."""

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

from .bpr import BPRRegularizedListwiseFM
from .contracts import TrialOutcome, ValidationMetrics
from .fingerprint import canonical_json
from .guards import evaluate_checked
from .history_pipeline import _load_baseline
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


def _save_checkpoint(path, model, fingerprint, trial_id, config, progress):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = canonical_json(config)
    payload = {
        **model.optimizer_state(),
        "dataset_fingerprint": np.asarray(fingerprint),
        "trial_id": np.asarray(trial_id),
        "config_json": np.asarray(config_json),
        "config_hash": np.asarray(hashlib.sha256(config_json.encode()).hexdigest()),
        "progress_json": np.asarray(canonical_json(progress)),
    }
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path, model, fingerprint, trial_id, config):
    config_json = canonical_json(config)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["dataset_fingerprint"]) != fingerprint:
            raise ValueError("BPR checkpoint fingerprint mismatch")
        if str(saved["trial_id"]) != trial_id:
            raise ValueError("BPR checkpoint trial mismatch")
        if str(saved["config_json"]) != config_json or str(saved["config_hash"]) != config_hash:
            raise ValueError("BPR checkpoint config mismatch")
        state = {key: saved[key].copy() for key in model.optimizer_state()}
        progress = json.loads(str(saved["progress_json"]))
    model.load_optimizer_state(state)
    return progress


def _train_bpr(trial, enc, baseline_model, groups, config, state_dir, fingerprint):
    hp = config["listwise_prior"]["hyperparameters"]
    weight = float(config["bpr_weight"])
    run_id = config["run_id"]
    artifact_dir = state_dir / "artifacts" / f"{trial['trial_id']}_{run_id}"
    latest_path, best_path = artifact_dir / "latest.npz", artifact_dir / "best.npz"
    model = BPRRegularizedListwiseFM(
        baseline_model,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_steps=int(hp["warmup_steps"]),
    )
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    rng = np.random.default_rng(0)
    progress = None
    if latest_path.exists():
        progress = _load_checkpoint(
            latest_path, model, fingerprint, trial["trial_id"], config
        )
        rng.bit_generator.state = progress["rng_state"]
    if progress is None:
        initial = evaluate_checked(uva, yva, model.predict(Xva))
        progress = {
            "epoch": 1,
            "next_group_start": 0,
            "order": [],
            "step": 0,
            "bad_checks": 0,
            "best_metrics": initial,
            "best_step": 0,
            "history_log": [
                {"step": 0, "epoch": 0.0, "train_loss": None, "validation": initial}
            ],
            "rng_state": rng.bit_generator.state,
        }
        _save_checkpoint(best_path, model, fingerprint, trial["trial_id"], config, progress)
        _save_checkpoint(latest_path, model, fingerprint, trial["trial_id"], config, progress)

    validation_interval = int(hp["validation_interval"])
    patience = int(hp["patience_checks"])
    batch_users = int(hp["user_batch_size"])
    max_epochs = int(hp["max_epochs"])
    epoch = int(progress["epoch"])
    step = int(progress["step"])
    bad_checks = int(progress["bad_checks"])
    best_metrics = dict(progress["best_metrics"])
    best_step = int(progress["best_step"])
    history_log = list(progress["history_log"])
    saved_order = np.asarray(progress["order"], dtype=np.int64)
    next_group_start = int(progress["next_group_start"])
    recent_losses = []
    stop_reason = "max_epochs"
    stop = False
    while epoch <= max_epochs and not stop:
        order = saved_order if len(saved_order) else rng.permutation(len(groups))
        start_offset = next_group_start if len(saved_order) else 0
        for group_start in range(start_offset, len(order), batch_users):
            batch = [groups[index] for index in order[group_start : group_start + batch_users]]
            row_indices = np.concatenate([group.row_indices for group in batch])
            sizes = [len(group.row_indices) for group in batch]
            weights = [group.positives for group in batch] if hp["metric_weighting"] else None
            recent_losses.append(
                model.step(
                    Xtr[row_indices],
                    ytr[row_indices],
                    sizes,
                    float(hp["score_temperature"]),
                    float(hp["target_temperature"]),
                    weights,
                    weight,
                    bool(hp["update_embeddings"]),
                )
            )
            step += 1
            if step % validation_interval:
                continue
            validation = evaluate_checked(uva, yva, model.predict(Xva))
            history_log.append(
                {
                    "step": step,
                    "epoch": epoch - 1 + min(1.0, (group_start + len(batch)) / len(groups)),
                    "train_loss": float(np.mean(recent_losses)),
                    "validation": validation,
                }
            )
            recent_losses.clear()
            if validation["primary"] > best_metrics["primary"]:
                best_metrics = validation
                best_step = step
                bad_checks = 0
                best_progress = {
                    "epoch": epoch,
                    "next_group_start": group_start + batch_users,
                    "order": order.tolist(),
                    "step": step,
                    "bad_checks": 0,
                    "best_metrics": best_metrics,
                    "best_step": best_step,
                    "history_log": history_log,
                    "rng_state": rng.bit_generator.state,
                }
                _save_checkpoint(
                    best_path, model, fingerprint, trial["trial_id"], config, best_progress
                )
            else:
                bad_checks += 1
            next_offset = group_start + batch_users
            next_epoch = epoch + 1 if next_offset >= len(order) else epoch
            latest_progress = {
                "epoch": next_epoch,
                "next_group_start": 0 if next_epoch != epoch else next_offset,
                "order": [] if next_epoch != epoch else order.tolist(),
                "step": step,
                "bad_checks": bad_checks,
                "best_metrics": best_metrics,
                "best_step": best_step,
                "history_log": history_log,
                "rng_state": rng.bit_generator.state,
            }
            _save_checkpoint(
                latest_path, model, fingerprint, trial["trial_id"], config, latest_progress
            )
            if bad_checks >= patience:
                stop_reason = "validation_patience"
                stop = True
                break
        epoch += 1
        saved_order = np.empty(0, dtype=np.int64)
        next_group_start = 0

    _load_checkpoint(best_path, model, fingerprint, trial["trial_id"], config)
    artifact = {
        "kind": "listnet_bpr_checkpoint",
        "path": str(best_path),
        "sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
    }
    return TrialOutcome(
        validation=ValidationMetrics.from_mapping(best_metrics),
        best_step=best_step,
        stop_reason=stop_reason,
        history=tuple(history_log),
        artifacts=(artifact,),
    )


def run_controlled(data_dir: Path, state_root: Path):
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
    controlled = json.loads(
        (Path(__file__).with_name("configs") / "bpr_weight_controlled.json").read_text()
    )
    h00 = store.get("trial-04")
    if h00["status"] != "completed" or h00["method"] != prior["method"]:
        raise RuntimeError("B-00 requires reusable no-history trial-04")
    train_scores = baseline_model.predict(enc["train"][0])
    groups = group_user_exposures(
        enc["train"][2],
        enc["train"][1],
        train_scores,
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    run_trials = {}
    for run_id, weight in controlled["runs"].items():
        config = {
            "schema_version": 1,
            "run_id": run_id,
            "method": controlled["method"],
            "seed": 0,
            "objective": controlled["objective"],
            "pair_scope": controlled["pair_scope"],
            "bpr_weight": float(weight),
            "listwise_prior": prior,
            "initialization": {
                "source": "fresh_validation_best_official_fm_on_current_dataset",
                "baseline_artifact_sha256": baseline_artifact["sha256"],
            },
        }
        matches = [
            trial
            for trial in store.trials()
            if trial["method"] == controlled["method"]
            and trial["config_hash"] == _config_hash(config)
        ]
        if matches and matches[0]["status"] in {"completed", "failed", "pruned"}:
            run_trials[run_id] = matches[0]
            continue
        resume_id = matches[0]["trial_id"] if matches else None
        spec = TrialSpec(
            controlled["method"],
            f"Controlled same-user BPR regularizer weight={weight}",
            config,
            0,
        )

        def candidate(trial, current_config=config):
            return _train_bpr(
                trial,
                enc,
                baseline_model,
                groups,
                current_config,
                state_dir,
                fingerprint,
            )

        run_trials[run_id] = runner.execute(spec, candidate, resume_trial_id=resume_id)

    base_primary = h00["validation"]["primary"]
    results = {
        "B-00": {
            "source_trial_id": h00["trial_id"],
            "bpr_weight": 0.0,
            "new_budget_cost": 0,
            "status": h00["status"],
            "validation": h00["validation"],
            "gain_vs_B-00": 0.0,
        }
    }
    for run_id, trial in run_trials.items():
        validation = trial.get("validation")
        results[run_id] = {
            "trial_id": trial["trial_id"],
            "bpr_weight": trial["config"]["bpr_weight"],
            "status": trial["status"],
            "validation": validation,
            "gain_vs_B-00": None if validation is None else validation["primary"] - base_primary,
        }
    write_top1(store, state_dir / "top1.json")
    payload = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "selection": "validation primary only",
        "test_metrics_used": False,
        "runs": results,
        "ledger": {"used": store.consumed, "remaining": store.remaining},
    }
    _atomic_json(state_dir / "bpr_weight_controlled_results.json", payload)
    return payload


def main():
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
