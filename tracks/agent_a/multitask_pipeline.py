"""M-00..M-04 validation-only multi-task controlled runs."""

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

from .auxiliary import load_training_auxiliary, read_training_identities
from .contracts import TrialOutcome, ValidationMetrics
from .fingerprint import canonical_json
from .guards import evaluate_checked
from .history_pipeline import _load_baseline
from .listwise import group_user_exposures
from .multitask_model import (
    MultiTaskListwiseFM,
    load_multitask_checkpoint,
    save_multitask_checkpoint,
)
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


def _train_multitask(
    trial: dict,
    enc: dict,
    baseline_model,
    groups,
    auxiliary,
    config: dict,
    state_dir: Path,
    fingerprint: str,
) -> TrialOutcome:
    hp = config["listwise_prior"]["hyperparameters"]
    aux_config = config["auxiliary"]
    artifact_dir = state_dir / "artifacts" / f"{trial['trial_id']}_{config['run_id']}"
    latest_path = artifact_dir / "latest.npz"
    best_path = artifact_dir / "best.npz"
    normalization = {
        "source": "train_only",
        "transform": "log1p_zscore",
        **auxiliary.play_transform.to_dict(),
    }
    model = MultiTaskListwiseFM(
        baseline_model,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_steps=int(hp["warmup_steps"]),
        click_weight=float(aux_config["is_click_weight"]),
        play_weight=float(aux_config["play_time_weight"]),
        head_lr=float(aux_config["head_lr"]),
        huber_delta=float(aux_config["huber_delta"]),
    )
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    clicks, play_targets = auxiliary.for_split("train")
    rng = np.random.default_rng(int(config["seed"]))
    progress = None
    if latest_path.exists():
        progress = load_multitask_checkpoint(
            latest_path, model, fingerprint, trial["trial_id"], config, normalization
        )
        rng.bit_generator.state = progress["rng_state"]
    if progress is None:
        initial = evaluate_checked(uva, yva, model.predict(Xva))
        history_log = [{"step": 0, "epoch": 0.0, "train_loss": None, "validation": initial}]
        progress = {
            "epoch": 1, "next_group_start": 0, "order": [], "step": 0,
            "bad_checks": 0, "best_metrics": initial, "best_step": 0,
            "history_log": history_log, "rng_state": rng.bit_generator.state,
        }
        save_multitask_checkpoint(
            best_path, model, fingerprint, trial["trial_id"], config,
            normalization, progress,
        )
        save_multitask_checkpoint(
            latest_path, model, fingerprint, trial["trial_id"], config,
            normalization, progress,
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
            losses = model.step(
                Xtr[row_indices], ytr[row_indices], clicks[row_indices], play_targets[row_indices],
                sizes, float(hp["score_temperature"]), float(hp["target_temperature"]),
                weights, bool(hp["update_embeddings"]),
            )
            recent_losses.append(losses["total"])
            step += 1
            if step % validation_interval:
                continue
            validation = evaluate_checked(uva, yva, model.predict(Xva))
            history_log.append({
                "step": step,
                "epoch": epoch - 1 + min(1.0, (group_start + len(batch)) / len(groups)),
                "train_loss": float(np.mean(recent_losses)),
                "validation": validation,
            })
            recent_losses.clear()
            if validation["primary"] > best_metrics["primary"]:
                best_metrics = validation
                best_step = step
                bad_checks = 0
                best_progress = {
                    "epoch": epoch, "next_group_start": group_start + batch_users,
                    "order": order.tolist(), "step": step, "bad_checks": bad_checks,
                    "best_metrics": best_metrics, "best_step": best_step,
                    "history_log": history_log, "rng_state": rng.bit_generator.state,
                }
                save_multitask_checkpoint(
                    best_path, model, fingerprint, trial["trial_id"], config,
                    normalization, best_progress,
                )
            else:
                bad_checks += 1
            next_offset = group_start + batch_users
            next_epoch = epoch + 1 if next_offset >= len(order) else epoch
            latest_progress = {
                "epoch": next_epoch,
                "next_group_start": 0 if next_epoch != epoch else next_offset,
                "order": [] if next_epoch != epoch else order.tolist(),
                "step": step, "bad_checks": bad_checks,
                "best_metrics": best_metrics, "best_step": best_step,
                "history_log": history_log, "rng_state": rng.bit_generator.state,
            }
            save_multitask_checkpoint(
                latest_path, model, fingerprint, trial["trial_id"], config,
                normalization, latest_progress,
            )
            if bad_checks >= patience:
                stop_reason = "validation_patience"
                stop = True
                break
        epoch += 1
        saved_order = np.empty(0, dtype=np.int64)
        next_group_start = 0

    load_multitask_checkpoint(
        best_path, model, fingerprint, trial["trial_id"], config, normalization
    )
    artifact = {
        "kind": "multitask_listwise_checkpoint",
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
    baseline_model, _, baseline_artifact = _load_baseline(store, dim, fingerprint)
    config_dir = Path(__file__).with_name("configs")
    prior = json.loads((config_dir / "listnet_prior.json").read_text())
    controlled = json.loads((config_dir / "multitask_controlled.json").read_text())
    m00 = store.get("trial-04")
    if m00["status"] != "completed" or m00["method"] != prior["method"]:
        raise RuntimeError("M-00 requires reusable no-history Listwise trial-04")
    identities = read_training_identities(data_dir)
    auxiliary = load_training_auxiliary(data_dir, splits, enc, identities)
    groups = group_user_exposures(
        enc["train"][2], enc["train"][1], baseline_model.predict(enc["train"][0]),
        int(prior["hyperparameters"]["hard_negative_cap"]),
    )
    runner = TrialRunner(store)
    run_trials = {}
    for run_id, weights in controlled["runs"].items():
        aux_config = {**controlled["fixed"], **weights}
        config = {
            "schema_version": 1, "run_id": run_id, "method": controlled["method"],
            "seed": 0, "listwise_prior": prior, "auxiliary": aux_config,
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
        spec = TrialSpec(controlled["method"], weights["description"], config, 0)

        def candidate(trial, current_config=config):
            return _train_multitask(
                trial, enc, baseline_model, groups, auxiliary,
                current_config, state_dir, fingerprint,
            )

        run_trials[run_id] = runner.execute(spec, candidate, resume_trial_id=resume_id)

    base_primary = m00["validation"]["primary"]
    results = {
        "M-00": {
            "source_trial_id": m00["trial_id"], "new_budget_cost": 0,
            "status": m00["status"], "validation": m00["validation"],
            "gain_vs_M-00": 0.0, "best_step": m00["result"]["best_step"],
        }
    }
    for run_id, trial in run_trials.items():
        validation = trial.get("validation")
        results[run_id] = {
            "trial_id": trial["trial_id"], "status": trial["status"],
            "validation": validation,
            "gain_vs_M-00": None if validation is None else validation["primary"] - base_primary,
            "best_step": None if trial.get("result") is None else trial["result"]["best_step"],
        }
    top1 = write_top1(store, state_dir / "top1.json")
    payload = {
        "schema_version": 1, "dataset_fingerprint": fingerprint,
        "objective": "long_view Soft-target ListNet with train-only auxiliary objectives",
        "selection": "validation long_view primary only", "test_metrics_used": False,
        "epsilon": 0.002, "runs": results, "validation_top1": top1,
        "ledger": {"used": store.consumed, "remaining": store.remaining},
    }
    _atomic_json(state_dir / "multitask_controlled_results.json", payload)
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
