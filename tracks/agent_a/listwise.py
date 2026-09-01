"""NumPy Soft-target ListNet fine-tuning over same-user logged exposures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExposureGroup:
    user_id: str
    row_indices: np.ndarray
    positives: int


def group_user_exposures(
    user_ids: Sequence[str],
    labels: np.ndarray,
    baseline_scores: np.ndarray | None = None,
    hard_negative_cap: int | None = None,
) -> list[ExposureGroup]:
    """Return discriminative, same-user groups with stable row membership."""
    if hard_negative_cap is not None and hard_negative_cap < 1:
        raise ValueError("hard_negative_cap must be at least one")
    if len(user_ids) != len(labels):
        raise ValueError("user_ids and labels must have identical lengths")
    if baseline_scores is not None and len(baseline_scores) != len(labels):
        raise ValueError("baseline scores must align with labels")
    rows_by_user: dict[str, list[int]] = {}
    for index, user_id in enumerate(user_ids):
        rows_by_user.setdefault(str(user_id), []).append(index)
    groups = []
    for user_id, rows in rows_by_user.items():
        positives = [row for row in rows if labels[row] == 1]
        negatives = [row for row in rows if labels[row] == 0]
        if not positives or not negatives:
            continue
        if hard_negative_cap is not None and hard_negative_cap >= 0 and len(negatives) > hard_negative_cap:
            if baseline_scores is None:
                raise ValueError("hard-negative selection requires baseline scores")
            negatives = sorted(negatives, key=lambda row: (-float(baseline_scores[row]), row))[
                :hard_negative_cap
            ]
        selected = np.asarray(sorted(positives + negatives), dtype=np.int64)
        groups.append(ExposureGroup(user_id, selected, len(positives)))
    return groups


def listnet_loss_and_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Iterable[int],
    score_temperature: float = 1.0,
    target_temperature: float = 0.5,
    group_weights: Iterable[float] | None = None,
) -> tuple[float, np.ndarray]:
    """Return weighted Soft-target ListNet cross-entropy and dL/dlogit."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    sizes = [int(size) for size in group_sizes]
    if score_temperature <= 0 or target_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if not sizes or sum(sizes) != len(logits) or len(labels) != len(logits):
        raise ValueError("group sizes must cover aligned logits and labels exactly")
    weights = np.ones(len(sizes), dtype=np.float64) if group_weights is None else np.asarray(
        list(group_weights), dtype=np.float64
    )
    if len(weights) != len(sizes) or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("group weights must be finite, positive, and aligned")
    weights /= weights.sum()
    gradient = np.empty_like(logits, dtype=np.float64)
    loss = 0.0
    start = 0
    for group_index, size in enumerate(sizes):
        stop = start + size
        relevance = labels[start:stop].astype(np.float64)
        positives = float(relevance.sum())
        if positives <= 0 or positives >= size:
            raise ValueError("ListNet accepts discriminative user groups only")
        score_scaled = logits[start:stop].astype(np.float64) / score_temperature
        score_shifted = score_scaled - np.max(score_scaled)
        probability = np.exp(score_shifted)
        normalizer = probability.sum()
        probability /= normalizer
        log_probability = score_shifted - np.log(normalizer)
        target_scaled = relevance / target_temperature
        target_shifted = target_scaled - np.max(target_scaled)
        target = np.exp(target_shifted)
        target /= target.sum()
        weight = float(weights[group_index])
        loss -= weight * float(np.dot(target, log_probability))
        gradient[start:stop] = weight * (probability - target) / score_temperature
        start = stop
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite ListNet loss or gradient")
    return loss, gradient


class ListwiseFM:
    """FM parameters initialized from a freshly trained current-dataset baseline."""

    def __init__(self, baseline, lr: float, weight_decay: float, warmup_steps: int):
        self.V = baseline.V.copy()
        self.W = baseline.W.copy()
        self.b = np.float32(baseline.b)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * ((summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2)))
        return self.b + self.W[X].sum(axis=1) + interactions, embeddings, summed

    def step(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        group_sizes: Sequence[int],
        score_temperature: float,
        target_temperature: float,
        group_weights: Sequence[float] | None,
        update_embeddings: bool = True,
    ) -> float:
        scores, embeddings, summed = self.logits(X)
        loss, score_gradient = listnet_loss_and_gradient(
            scores, labels, group_sizes, score_temperature, target_temperature, group_weights
        )
        score_gradient = score_gradient.astype(np.float32)
        grad_w = np.zeros_like(self.W)
        grad_v = np.zeros_like(self.V)
        np.add.at(grad_w, X, score_gradient[:, None])
        np.add.at(
            grad_v,
            X,
            score_gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        self.t += 1
        rate = self.lr * (min(1.0, self.t / self.warmup_steps) if self.warmup_steps else 1.0)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first, second in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            if parameter is self.V and not update_embeddings:
                continue
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * gradient * gradient
            first_hat = first / (1 - beta1**self.t)
            second_hat = second / (1 - beta2**self.t)
            parameter *= 1.0 - rate * self.weight_decay
            parameter -= rate * first_hat / (np.sqrt(second_hat) + epsilon)
        return float(loss)

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [self.logits(X[start : start + batch_size])[0] for start in range(0, len(X), batch_size)]
        )

    def state(self) -> tuple[np.ndarray, np.ndarray, np.float32]:
        return self.V.copy(), self.W.copy(), np.float32(self.b)

    def load_state(self, state) -> None:
        self.V, self.W, self.b = state[0].copy(), state[1].copy(), np.float32(state[2])
