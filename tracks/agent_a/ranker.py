"""LightGBM LambdaRank and FM-ranker ensemble candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .behavioral_features import BehavioralFeatureBundle
from .contracts import TrialOutcome, ValidationMetrics
from .guards import evaluate_checked


def group_sorted_rows(users: list[str] | tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(users, dtype=str)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    if len(sorted_values) == 0:
        return order, np.empty(0, dtype=np.int32)
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    groups = np.diff(np.concatenate(([0], boundaries, [len(values)]))).astype(np.int32)
    return order, groups


def normalize_within_user(scores: np.ndarray, users: list[str] | tuple[str, ...]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(users),) or not np.all(np.isfinite(scores)):
        raise ValueError("user normalization requires one finite score per exposure")
    output = np.zeros_like(scores)
    by_user: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        by_user.setdefault(str(user), []).append(index)
    for indices in by_user.values():
        index = np.asarray(indices, dtype=np.int64)
        values = scores[index]
        scale = float(np.std(values))
        output[index] = values - float(np.mean(values))
        if scale > 1e-12:
            output[index] /= scale
    return output


def train_ranker_candidate(
    enc: dict,
    features: BehavioralFeatureBundle,
    baseline_model,
    artifact_path: Path,
    fingerprint: str,
    *,
    ensemble: bool,
    seed: int = 0,
    n_estimators: int = 160,
    learning_rate: float = 0.04,
    num_leaves: int = 31,
    min_child_samples: int = 50,
    validation_interval: int = 20,
) -> tuple[lgb.Booster, TrialOutcome]:
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    if len(features.train) != len(ytr) or len(features.evaluation) != len(yva):
        raise ValueError("ranker features are not row-aligned")
    order, groups = group_sorted_rows(features.train_users)
    if len(groups) == 0:
        raise ValueError("ranker requires at least one user group")
    dataset = lgb.Dataset(
        features.train[order], label=ytr[order], group=groups,
        categorical_feature=list(features.categorical_indices),
        free_raw_data=False,
    )
    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "None",
            "label_gain": [0, 1],
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "min_data_in_leaf": int(min_child_samples),
            "feature_fraction": 0.9,
            "lambda_l2": 1.0,
            "seed": int(seed),
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=int(n_estimators),
    )
    baseline_scores = np.asarray(baseline_model.predict(Xva), dtype=np.float64)
    baseline_normalized = normalize_within_user(baseline_scores, uva)
    alphas = (1.0,) if not ensemble else (0.25, 0.5, 0.75, 1.0)
    best_metrics = None
    best_iteration = 0
    best_alpha = 1.0
    history = []
    checkpoints = list(range(validation_interval, n_estimators + 1, validation_interval))
    if not checkpoints or checkpoints[-1] != n_estimators:
        checkpoints.append(n_estimators)
    for iteration in checkpoints:
        raw = np.asarray(model.predict(features.evaluation, num_iteration=iteration), dtype=np.float64)
        ranker_normalized = normalize_within_user(raw, uva)
        for alpha in alphas:
            scores = alpha * ranker_normalized + (1.0 - alpha) * baseline_normalized
            metrics = evaluate_checked(uva, yva, scores)
            history.append({
                "iteration": iteration,
                "ranker_weight": alpha,
                "validation": metrics,
            })
            if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
                best_metrics = metrics
                best_iteration = iteration
                best_alpha = float(alpha)
    model_text = model.model_to_string(num_iteration=best_iteration)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        model_text=np.asarray(model_text),
        best_iteration=np.asarray(best_iteration),
        ranker_weight=np.asarray(best_alpha),
        feature_schema_sha256=np.asarray(features.schema_sha256),
        dataset_fingerprint=np.asarray(fingerprint),
        seed=np.asarray(seed),
        V=baseline_model.V,
        W=baseline_model.W,
        b=baseline_model.b,
    )
    artifact = {
        "kind": "fm_lambdarank_ensemble_checkpoint" if ensemble else "lambdarank_checkpoint",
        "path": str(artifact_path),
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    outcome = TrialOutcome(
        ValidationMetrics.from_mapping(best_metrics),
        best_iteration,
        "validation_primary_checkpoint_selection",
        tuple(history),
        (artifact,),
    )
    return model, outcome


def predict_ranker_checkpoint(
    checkpoint,
    features: np.ndarray,
    encoded: np.ndarray,
    users: list[str] | tuple[str, ...],
) -> np.ndarray:
    booster = lgb.Booster(model_str=str(checkpoint["model_text"]))
    iteration = int(checkpoint["best_iteration"])
    ranker = normalize_within_user(booster.predict(features, num_iteration=iteration), users)
    weight = float(checkpoint["ranker_weight"])
    if weight >= 1.0:
        return ranker
    V, W, b = checkpoint["V"], checkpoint["W"], float(checkpoint["b"])
    embeddings = V[encoded]
    summed = embeddings.sum(axis=1)
    fm = b + W[encoded].sum(axis=1) + 0.5 * (
        (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
    )
    return weight * ranker + (1.0 - weight) * normalize_within_user(fm, users)
