"""Executable baseline -> ListNet prior -> validation Top-1 development pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import baseline as official_baseline
from data import encode, load

from .contracts import TrialOutcome, ValidationMetrics
from .guards import evaluate_checked
from .listwise import ListwiseFM, group_user_exposures
from .onboarding import onboard
from .runner import TrialRunner, TrialSpec
from .selection import write_top1
from .store import ResearchStore


def _artifact(path: Path, kind: str) -> dict:
    return {
        "kind": kind,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _metrics(mapping: dict) -> ValidationMetrics:
    return ValidationMetrics.from_mapping(mapping)


def train_baseline_candidate(enc, dim: int, artifact_path: Path, fingerprint: str):
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    model = official_baseline.FM(dim, k=16, lr=0.001, seed=0)
    rng = np.random.default_rng(0)
    best_metrics = None
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    for epoch in range(1, 41):
        permutation = rng.permutation(len(ytr))
        losses = []
        for start in range(0, len(permutation), 8192):
            index = permutation[start : start + 8192]
            losses.append(model.step(Xtr[index], ytr[index]))
        validation = evaluate_checked(uva, yva, model.predict(Xva))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation})
        if best_metrics is None or validation["primary"] > best_metrics["primary"] + 1e-5:
            best_metrics = validation
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.V, model.W, model.b = best_state
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        V=model.V,
        W=model.W,
        b=model.b,
        dataset_fingerprint=np.asarray(fingerprint),
        seed=np.asarray(0),
    )
    outcome = TrialOutcome(
        validation=_metrics(best_metrics),
        best_step=best_epoch,
        stop_reason="official_fm_validation_patience",
        history=tuple(history),
        artifacts=(_artifact(artifact_path, "fresh_baseline_checkpoint"),),
    )
    return model, outcome


def train_listnet_candidate(
    enc,
    baseline_model,
    config: dict,
    artifact_path: Path,
    fingerprint: str,
) -> tuple[ListwiseFM, TrialOutcome]:
    hp = config["hyperparameters"]
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    baseline_train_scores = baseline_model.predict(Xtr)
    groups = group_user_exposures(
        utr,
        ytr,
        baseline_scores=baseline_train_scores,
        hard_negative_cap=int(hp["hard_negative_cap"]),
    )
    if not groups:
        raise RuntimeError("no discriminative training users")
    model = ListwiseFM(
        baseline_model,
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_steps=int(hp["warmup_steps"]),
    )
    initial = evaluate_checked(uva, yva, model.predict(Xva))
    best_metrics = initial
    best_state = model.state()
    best_step = 0
    history = [{"step": 0, "epoch": 0.0, "train_loss": None, "validation": initial}]
    rng = np.random.default_rng(int(config["seed"]))
    validation_interval = int(hp["validation_interval"])
    patience_checks = int(hp["patience_checks"])
    batch_users = int(hp["user_batch_size"])
    bad_checks = 0
    stop_reason = "max_epochs"
    step = 0
    stop = False
    recent_losses = []
    for epoch in range(1, int(hp["max_epochs"]) + 1):
        order = rng.permutation(len(groups))
        for group_start in range(0, len(order), batch_users):
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
                    bool(hp["update_embeddings"]),
                )
            )
            step += 1
            if step % validation_interval:
                continue
            validation = evaluate_checked(uva, yva, model.predict(Xva))
            history.append(
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
                best_state = model.state()
                best_step = step
                bad_checks = 0
            else:
                bad_checks += 1
                if bad_checks >= patience_checks:
                    stop_reason = "validation_patience"
                    stop = True
                    break
        if stop:
            break
    model.load_state(best_state)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        V=model.V,
        W=model.W,
        b=model.b,
        dataset_fingerprint=np.asarray(fingerprint),
        seed=np.asarray(int(config["seed"])),
        best_step=np.asarray(best_step),
    )
    outcome = TrialOutcome(
        validation=_metrics(best_metrics),
        best_step=best_step,
        stop_reason=stop_reason,
        history=tuple(history),
        artifacts=(_artifact(artifact_path, "current_dataset_listnet_checkpoint"),),
    )
    return model, outcome


def run_development(data_dir: Path, state_root: Path) -> dict:
    onboarding = onboard(data_dir, state_root, purpose="development")
    manifest = onboarding["manifest"]
    fingerprint = manifest["dataset_fingerprint"]
    state_dir = Path(onboarding["state_dir"])
    store = ResearchStore(Path(onboarding["ledger"]), fingerprint)
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    runner = TrialRunner(store)
    baseline_holder = {}

    baseline_spec = TrialSpec(
        method="official_fm_baseline_reproduction",
        hypothesis="Reproduce the official FM on this dataset using validation-only early stopping.",
        config={"k": 16, "lr": 0.001, "batch_size": 8192, "max_epochs": 40, "patience": 4},
    )

    def baseline_run(_trial):
        model, outcome = train_baseline_candidate(
            enc, dim, state_dir / "artifacts" / "baseline_seed0.npz", fingerprint
        )
        baseline_holder["model"] = model
        return outcome

    completed_baselines = [
        trial
        for trial in store.trials()
        if trial["status"] == "completed" and trial["method"] == baseline_spec.method
    ]
    if completed_baselines:
        artifact = completed_baselines[0]["result"]["artifacts"][0]
        if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("baseline checkpoint digest mismatch")
        with np.load(artifact["path"]) as checkpoint:
            saved_fingerprint = str(checkpoint["dataset_fingerprint"])
            if saved_fingerprint != fingerprint:
                raise ValueError("baseline checkpoint fingerprint mismatch")
            model = official_baseline.FM(dim, k=16, lr=0.001, seed=0)
            model.V = checkpoint["V"].copy()
            model.W = checkpoint["W"].copy()
            model.b = np.float32(checkpoint["b"].item())
        baseline_holder["model"] = model
    else:
        runner.execute(baseline_spec, baseline_run)
    prior_path = Path(__file__).with_name("configs") / "listnet_prior.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    listwise_spec = TrialSpec(
        method=prior["method"],
        hypothesis="Test the predecessor Soft-target ListNet recipe as a prior on this dataset.",
        config=prior,
        seed=int(prior["seed"]),
    )

    def listwise_run(_trial):
        _, outcome = train_listnet_candidate(
            enc,
            baseline_holder["model"],
            prior,
            state_dir / "artifacts" / "listnet_prior_seed0.npz",
            fingerprint,
        )
        return outcome

    completed_listwise = [
        trial
        for trial in store.trials()
        if trial["status"] == "completed" and trial["method"] == listwise_spec.method
    ]
    if not completed_listwise:
        runner.execute(listwise_spec, listwise_run)
    top1 = write_top1(store, state_dir / "top1.json")
    return {
        "dataset_fingerprint": fingerprint,
        "state_dir": str(state_dir),
        "consumed_trials": store.consumed,
        "remaining_trials": store.remaining,
        "top1": top1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="KuaiRand-Pure/data")
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    args = parser.parse_args()
    started = time.time()
    result = run_development(Path(args.data_dir), Path(args.state_root))
    result["wall_time_sec"] = time.time() - started
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
