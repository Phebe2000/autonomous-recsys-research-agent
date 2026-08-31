"""Score locked exposure rows without evaluating or consulting held-out labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from data import SPLITS
from submit import write_submission

from .fingerprint import fingerprint_dataset, fingerprint_judged_dataset
from .codegen import load_generated_module
from .compliance import JudgedRunCompliance, JudgedRunPolicy
from .history_model import history_residual_and_gradients
from .behavioral_features import build_behavioral_features
from .ranker import predict_ranker_checkpoint
from .store import ResearchStore
from .safe_data import (
    encode_research_splits,
    load_research_splits,
    load_safe_side_features,
    load_unlabeled_exposures,
)


def _raw_identities(data_dir: Path, split: str) -> list[tuple[int, str, str, int]]:
    low, high = SPLITS[split]
    rows = []
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (Path(data_dir) / filename).open(newline="") as stream:
            for row in csv.DictReader(stream):
                date = int(row["date"])
                if low <= date <= high:
                    rows.append((date, row["user_id"], row["video_id"], int(row["time_ms"])))
    return rows


def frozen_train_positive_history_csr(
    data_dir: Path,
    splits: dict,
    enc: dict,
    split: str,
    row_indices: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use only train-positive rows as frozen inference history."""
    target_users, frozen = _frozen_train_positive_index(data_dir, splits, enc, split)
    return _history_csr_from_index(target_users, frozen, row_indices, limit)


def _frozen_train_positive_index(
    data_dir: Path,
    splits: dict,
    enc: dict,
    split: str,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Build the frozen train-positive index once for batched scoring."""
    if split not in {"valid", "test"}:
        raise ValueError("locked inference history is available only for valid/test exposures")
    train_raw = _raw_identities(data_dir, "train")
    target_raw = _raw_identities(data_dir, split)
    if len(train_raw) != len(splits["train"]) or len(target_raw) != len(splits[split]):
        raise ValueError("locked inference history raw row count mismatch")
    for name, raw, official in (
        ("train", train_raw, splits["train"]),
        (split, target_raw, splits[split]),
    ):
        for index, (identity, row) in enumerate(zip(raw, official)):
            if identity[:3] != (int(row[0]), str(row[1]), str(row[2])):
                raise ValueError(f"locked inference identity mismatch at {name} row {index}")
    items_by_user: dict[str, list[tuple[int, int, int]]] = {}
    train_items = enc["train"][0][:, 1]
    for index, (identity, official, item) in enumerate(zip(train_raw, splits["train"], train_items)):
        if int(official[6]) == 1:
            items_by_user.setdefault(identity[1], []).append((identity[3], index, int(item)))
    frozen = {
        user: np.asarray(
            [item for _, _, item in sorted(events, key=lambda event: (event[0], event[1]))],
            dtype=np.int32,
        )
        for user, events in items_by_user.items()
    }
    return [identity[1] for identity in target_raw], frozen


def _history_csr_from_index(
    target_users: list[str],
    frozen: dict[str, np.ndarray],
    row_indices: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is not None and limit <= 0:
        raise ValueError("history limit must be positive or null")
    indices = np.asarray(row_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(target_users)):
        raise IndexError("locked inference exposure index out of range")
    indptr = np.zeros(len(indices) + 1, dtype=np.int64)
    chunks = []
    for output_index, row_index in enumerate(indices):
        items = frozen.get(target_users[int(row_index)], np.empty(0, dtype=np.int32))
        selected = items if limit is None else items[-limit:]
        chunks.append(selected)
        indptr[output_index + 1] = indptr[output_index] + len(selected)
    flat = np.concatenate(chunks) if chunks and indptr[-1] else np.empty(0, dtype=np.int32)
    return indptr, flat


def _fm_scores(X: np.ndarray, V: np.ndarray, W: np.ndarray, b: float) -> np.ndarray:
    embeddings = V[X]
    summed = embeddings.sum(axis=1)
    interaction = 0.5 * (
        (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
    )
    return np.asarray(b, dtype=np.float32) + W[X].sum(axis=1) + interaction


def score_locked_exposures(
    data_dir: Path,
    state_root: Path,
    output_path: Path,
    split: str = "test",
    judged: bool = False,
) -> dict:
    if split not in {"valid", "test"}:
        raise ValueError("locked scoring supports valid or test exposure rows")
    fingerprint_fn = fingerprint_judged_dataset if judged else fingerprint_dataset
    fingerprint = fingerprint_fn(data_dir)["dataset_fingerprint"]
    state_dir = Path(state_root) / fingerprint.removeprefix("sha256:")
    lock_path = state_dir / "locked_manifest.json"
    if not lock_path.exists():
        raise FileNotFoundError("validation Top-1 must be locked before scoring exposures")
    manifest = json.loads(lock_path.read_text())
    if manifest.get("simulation") or not manifest.get("production_top1_eligible"):
        raise ValueError("simulation locks cannot produce production exposure scores")
    if manifest["dataset_fingerprint"] != fingerprint:
        raise ValueError("lock/dataset fingerprint mismatch")
    store = ResearchStore(state_dir / "research.sqlite3", fingerprint)
    best = store.best_trial()
    if best is None or best["trial_id"] != manifest["trial_id"]:
        raise ValueError("locked trial no longer matches validation-only Top-1")
    artifacts = best["result"]["artifacts"]
    if not artifacts:
        raise ValueError("locked trial has no checkpoint artifact")
    artifact = artifacts[0]
    checkpoint_path = Path(artifact["path"])
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("locked checkpoint digest mismatch")

    research_splits = load_research_splits(data_dir)
    enc, encoder = encode_research_splits(research_splits)
    exposures = load_unlabeled_exposures(data_dir, split)
    X = encoder.transform(exposures)
    output_rows = [exposure.submission_row() for exposure in exposures]
    splits = {"train": research_splits["train"], split: output_rows}
    config = best["config"].get("unified_candidate", {}).get("candidate", {})
    history_config = config.get("history", {"enabled": False})
    outputs = []
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        if artifact["kind"] in {
            "lambdarank_checkpoint", "fm_lambdarank_ensemble_checkpoint",
        }:
            ranker_enc = dict(enc)
            ranker_enc[split] = (X, None, [row.user_id for row in exposures])
            features = build_behavioral_features(
                data_dir, ranker_enc, encoder, split,
                float(config["ranker"]["smoothing"]),
            )
            if features.schema_sha256 != str(checkpoint["feature_schema_sha256"]):
                raise ValueError("ranker feature schema/checkpoint mismatch")
            outputs.append(predict_ranker_checkpoint(
                checkpoint, features.evaluation, X, ranker_enc[split][2]
            ))
            history_config = {"enabled": False}
            V = W = None
            b = 0.0
        elif artifact["kind"] == "generated_candidate_checkpoint":
            source_artifacts = [
                item for item in artifacts if item["kind"] == "generated_candidate_source"
            ]
            if len(source_artifacts) != 1:
                raise ValueError("generated checkpoint requires exactly one source artifact")
            source_artifact = source_artifacts[0]
            source_path = Path(source_artifact["path"])
            source = source_path.read_text()
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_artifact["sha256"]:
                raise ValueError("generated source digest mismatch")
            if hashlib.sha256(source.encode()).hexdigest() != str(checkpoint["source_sha256"]):
                raise ValueError("generated checkpoint/source mismatch")
            module = load_generated_module(source)
            side = load_safe_side_features(data_dir, split)
            state = {
                key.removeprefix("state_"): checkpoint[key]
                for key in checkpoint.files
                if key.startswith("state_")
            }
            for start in range(0, len(X), 100_000):
                stop = min(len(X), start + 100_000)
                batch_side = {key: value[start:stop] for key, value in side.items()}
                scores = np.asarray(
                    module["predict"](X[start:stop], batch_side, state),
                    dtype=np.float64,
                )
                if scores.shape != (stop - start,):
                    raise ValueError("generated final predictor returned the wrong row count")
                outputs.append(scores)
            history_config = {"enabled": False}
            V = W = None
            b = 0.0
        else:
            V, W = checkpoint["V"], checkpoint["W"]
            b = float(checkpoint["b"])
        H = checkpoint["H"] if history_config.get("enabled") else None
        gate = float(checkpoint["gate"]) if history_config.get("enabled") else None
        history_index = (
            _frozen_train_positive_index(data_dir, splits, enc, split)
            if history_config.get("enabled") else None
        )
        for start in range(0 if V is not None else len(X), len(X), 100_000):
            stop = min(len(X), start + 100_000)
            scores = _fm_scores(X[start:stop], V, W, b)
            if history_config.get("enabled"):
                rows = np.arange(start, stop, dtype=np.int64)
                indptr, history_ids = _history_csr_from_index(
                    history_index[0], history_index[1], rows, history_config["last_n"]
                )
                residual = history_residual_and_gradients(
                    X[start:stop, 1], indptr, history_ids, H, gate
                )[0]
                scores = scores + residual.astype(scores.dtype, copy=False)
            outputs.append(scores)
    predictions = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    if len(predictions) != len(output_rows) or not np.all(np.isfinite(predictions)):
        raise ValueError("locked exposure scores are not finite and row-aligned")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output_path, output_rows, predictions)
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    store.record_agent_decision(
        stage="final_inference",
        decision="emit_locked_exposure_scores",
        rationale="Generate row-aligned scores from the immutable validation-selected checkpoint without loading labels or computing metrics.",
        evidence={
            "split": split,
            "rows": len(predictions),
            "locked_trial_id": best["trial_id"],
            "checkpoint_sha256": artifact["sha256"],
            "output_sha256": output_sha256,
            "metrics_computed": False,
            "hidden_labels_loaded": False,
        },
        alternatives=("change_checkpoint_after_lock", "evaluate_labels_before_submission"),
        selected_action="write_official_submission_schema",
        actor="agent-a-finalizer",
        trial_id=best["trial_id"],
        decision_key=f"final-inference:{split}:{output_sha256}",
        data_scope="label_free_locked_inference",
    )
    if judged:
        policy_data = json.loads((state_dir / "run_policy.json").read_text())
        policy_data.pop("policy_identity", None)
        JudgedRunCompliance(
            state_dir, JudgedRunPolicy(**policy_data)
        ).write_audit(store)
    result = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "locked_trial_id": best["trial_id"],
        "split": split,
        "rows": len(predictions),
        "output": str(output_path),
        "checkpoint_sha256": artifact["sha256"],
        "output_sha256": output_sha256,
        "selection": "pre-existing validation-only lock",
        "metrics_computed": False,
        "test_metrics_used": False,
        "hidden_labels_loaded": False,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--judged", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        score_locked_exposures(
            Path(args.data_dir), Path(args.state_root), Path(args.output), args.split, args.judged
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
