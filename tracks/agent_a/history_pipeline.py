"""Validation-only H-00..H-04 controlled history runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

import baseline as official_baseline
from data import encode, load

from .contracts import TrialOutcome, ValidationMetrics
from .fingerprint import canonical_json
from .guards import evaluate_checked
from .history import build_causal_history
from .history_model import (
    HistoryListwiseFM,
    load_history_checkpoint,
    save_history_checkpoint,
)
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


def _load_baseline(store: ResearchStore, dim: int, fingerprint: str):
    matches = [
        trial
        for trial in store.trials()
        if trial["status"] == "completed" and trial["method"] == "official_fm_baseline_reproduction"
    ]
    if not matches:
        raise RuntimeError("fresh current-dataset baseline trial is required")
    trial = matches[0]
    artifact = trial["result"]["artifacts"][0]
    path = Path(artifact["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("baseline checkpoint digest mismatch")
    model = official_baseline.FM(dim, k=16, lr=0.001, seed=0)
    with np.load(path, allow_pickle=False) as checkpoint:
        if str(checkpoint["dataset_fingerprint"]) != fingerprint:
            raise ValueError("baseline checkpoint fingerprint mismatch")
        model.V = checkpoint["V"].copy()
        model.W = checkpoint["W"].copy()
        model.b = np.float32(checkpoint["b"].item())
    return model, trial, artifact


def _history_config(run_id: str, limit: int | None, prior: dict, controlled: dict, baseline_artifact: dict):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "method": controlled["method"],
        "seed": 0,
        "listwise_prior": prior,
        "history": {**controlled["history"], "last_n": limit},
        "initialization": {
            "source": "fresh_validation_best_official_fm_on_current_dataset",
            "baseline_artifact_sha256": baseline_artifact["sha256"],
        },
    }


def _train_history(
    trial: dict,
    enc: dict,
    baseline_model,
    groups,
    history_index,
    config: dict,
    state_dir: Path,
    fingerprint: str,
) -> TrialOutcome:
    run_id = config["run_id"]
    limit = config["history"]["last_n"]
    hp = config["listwise_prior"]["hyperparameters"]
    artifact_dir = state_dir / "artifacts" / f"{trial['trial_id']}_{run_id}"
    latest_path = artifact_dir / "latest.npz"
    best_path = artifact_dir / "best.npz"
    model = HistoryListwiseFM(
        baseline_model,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_steps=int(hp["warmup_steps"]),
        history_dim=int(config["history"]["history_dim"]),
        gate=float(config["history"]["history_gate_initial"]),
        history_lr=config["history"].get("history_lr"),
        gate_lr=config["history"].get("gate_lr"),
        history_unfreeze_step=int(config["history"].get("history_unfreeze_step", 0)),
        train_history_embeddings=bool(config["history"].get("train_history_embeddings", True)),
        train_gate=bool(config["history"].get("train_gate", True)),
    )
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    rng = np.random.default_rng(int(config.get("seed", 0)))
    progress = None
    if latest_path.exists():
        progress = load_history_checkpoint(
            latest_path, model, fingerprint, trial["trial_id"], config
        )
        rng.bit_generator.state = progress["rng_state"]
    if progress is None:
        initial = evaluate_checked(
            uva, yva, model.predict(Xva, history_index, "valid", limit)
        )
        history_log = [
            {"step": 0, "epoch": 0.0, "train_loss": None, "validation": initial}
        ]
        progress = {
            "epoch": 1,
            "next_group_start": 0,
            "order": [],
            "step": 0,
            "bad_checks": 0,
            "best_metrics": initial,
            "best_step": 0,
            "history_log": history_log,
            "rng_state": rng.bit_generator.state,
        }
        save_history_checkpoint(
            best_path, model, fingerprint, trial["trial_id"], config, progress
        )
        save_history_checkpoint(
            latest_path, model, fingerprint, trial["trial_id"], config, progress
        )

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
            indptr, history_ids = history_index.batch_csr("train", row_indices, limit)
            recent_losses.append(
                model.step(
                    Xtr[row_indices],
                    ytr[row_indices],
                    sizes,
                    indptr,
                    history_ids,
                    float(hp["score_temperature"]),
                    float(hp["target_temperature"]),
                    weights,
                    bool(hp["update_embeddings"]),
                )
            )
            step += 1
            if step % validation_interval:
                continue
            validation = evaluate_checked(
                uva, yva, model.predict(Xva, history_index, "valid", limit)
            )
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
                    "bad_checks": bad_checks,
                    "best_metrics": best_metrics,
                    "best_step": best_step,
                    "history_log": history_log,
                    "rng_state": rng.bit_generator.state,
                }
                save_history_checkpoint(
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
            save_history_checkpoint(
                latest_path, model, fingerprint, trial["trial_id"], config, latest_progress
            )
            if bad_checks >= patience:
                stop_reason = "validation_patience"
                stop = True
                break
        epoch += 1
        saved_order = np.empty(0, dtype=np.int64)
        next_group_start = 0

    best_progress = load_history_checkpoint(
        best_path, model, fingerprint, trial["trial_id"], config
    )
    artifact = {
        "kind": "causal_positive_history_checkpoint",
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


def run_controlled(data_dir: Path, state_root: Path) -> dict:
    onboarding = onboard(data_dir, state_root, purpose="development")
    fingerprint = onboarding["manifest"]["dataset_fingerprint"]
    state_dir = Path(onboarding["state_dir"])
    store = ResearchStore(Path(onboarding["ledger"]), fingerprint)
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    baseline_model, baseline_trial, baseline_artifact = _load_baseline(store, dim, fingerprint)
    prior = json.loads(
        (Path(__file__).with_name("configs") / "listnet_prior.json").read_text()
    )
    controlled = json.loads(
        (Path(__file__).with_name("configs") / "history_controlled.json").read_text()
    )
    h00_matches = [
        trial
        for trial in store.trials()
        if trial["trial_id"] == "trial-04"
        and trial["status"] == "completed"
        and trial["method"] == prior["method"]
    ]
    if not h00_matches:
        raise RuntimeError("H-00 requires reusable Milestone 1 trial-04")
    h00 = h00_matches[0]
    history_index = build_causal_history(data_dir, splits, enc)
    baseline_train_scores = baseline_model.predict(enc["train"][0])
    groups = group_user_exposures(
        enc["train"][2],
        enc["train"][1],
        baseline_train_scores,
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    run_trials = {}
    for run_id, limit in controlled["runs"].items():
        config = _history_config(run_id, limit, prior, controlled, baseline_artifact)
        config_hash = _config_hash(config)
        matches = [
            trial
            for trial in store.trials()
            if trial["method"] == controlled["method"] and trial["config_hash"] == config_hash
        ]
        if matches:
            existing = matches[0]
            if existing["status"] == "completed":
                run_trials[run_id] = existing
                continue
            if existing["status"] in {"failed", "pruned"}:
                run_trials[run_id] = existing
                continue
            resume_id = existing["trial_id"]
        else:
            resume_id = None
        spec = TrialSpec(
            method=controlled["method"],
            hypothesis=f"Controlled causal positive-history mean pooling with last_n={limit}",
            config=config,
            seed=0,
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

    h00_primary = h00["validation"]["primary"]
    results = {
        "H-00": {
            "source_trial_id": h00["trial_id"],
            "new_budget_cost": 0,
            "status": h00["status"],
            "validation": h00["validation"],
            "gain_vs_H-00": 0.0,
        }
    }
    for run_id, trial in run_trials.items():
        validation = trial.get("validation")
        results[run_id] = {
            "trial_id": trial["trial_id"],
            "status": trial["status"],
            "validation": validation,
            "gain_vs_H-00": None if validation is None else validation["primary"] - h00_primary,
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
    _atomic_json(state_dir / "history_controlled_results.json", payload)
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
