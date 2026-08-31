"""Same-user BPR regularization layered on the validation-only ListNet model."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .listwise import ListwiseFM, listnet_loss_and_gradient


def bpr_loss_and_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Iterable[int],
    group_weights: Iterable[float] | None = None,
) -> tuple[float, np.ndarray]:
    """Average all positive-negative pairs inside each user group."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    sizes = [int(size) for size in group_sizes]
    if not sizes or sum(sizes) != len(logits) or len(labels) != len(logits):
        raise ValueError("group sizes must cover aligned logits and labels exactly")
    weights = np.ones(len(sizes), dtype=np.float64) if group_weights is None else np.asarray(
        list(group_weights), dtype=np.float64
    )
    if len(weights) != len(sizes) or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("group weights must be finite, positive, and aligned")
    weights /= weights.sum()
    gradient = np.zeros_like(logits, dtype=np.float64)
    loss = 0.0
    start = 0
    for group_index, size in enumerate(sizes):
        stop = start + size
        relevance = labels[start:stop]
        positives = np.flatnonzero(relevance == 1)
        negatives = np.flatnonzero(relevance == 0)
        if not len(positives) or not len(negatives):
            raise ValueError("BPR accepts discriminative same-user groups only")
        differences = logits[start:stop][positives, None] - logits[start:stop][None, negatives]
        pair_loss = np.logaddexp(0.0, -differences)
        weight = float(weights[group_index])
        loss += weight * float(pair_loss.mean())
        # sigmoid(-difference), evaluated stably.
        pair_gradient = np.exp(-np.logaddexp(0.0, differences))
        scale = weight / pair_gradient.size
        np.add.at(gradient, start + positives, -scale * pair_gradient.sum(axis=1))
        np.add.at(gradient, start + negatives, scale * pair_gradient.sum(axis=0))
        start = stop
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite BPR loss or gradient")
    return loss, gradient


def listnet_bpr_loss_and_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Sequence[int],
    bpr_weight: float,
    score_temperature: float,
    target_temperature: float,
    group_weights: Sequence[float] | None,
) -> tuple[float, np.ndarray]:
    if bpr_weight < 0:
        raise ValueError("bpr_weight must be non-negative")
    listnet_loss, listnet_gradient = listnet_loss_and_gradient(
        logits,
        labels,
        group_sizes,
        score_temperature,
        target_temperature,
        group_weights,
    )
    if bpr_weight == 0.0:
        return listnet_loss, listnet_gradient
    bpr_loss, bpr_gradient = bpr_loss_and_gradient(logits, labels, group_sizes, group_weights)
    return listnet_loss + bpr_weight * bpr_loss, listnet_gradient + bpr_weight * bpr_gradient


class BPRRegularizedListwiseFM(ListwiseFM):
    def step(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        group_sizes: Sequence[int],
        score_temperature: float,
        target_temperature: float,
        group_weights: Sequence[float] | None,
        bpr_weight: float,
        update_embeddings: bool = True,
    ) -> float:
        scores, embeddings, summed = self.logits(X)
        loss, score_gradient = listnet_bpr_loss_and_gradient(
            scores,
            labels,
            group_sizes,
            bpr_weight,
            score_temperature,
            target_temperature,
            group_weights,
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
            parameter *= 1.0 - rate * self.weight_decay
            parameter -= rate * (first / (1 - beta1**self.t)) / (
                np.sqrt(second / (1 - beta2**self.t)) + epsilon
            )
        return float(loss)

    def optimizer_state(self) -> dict:
        return {
            "V": self.V.copy(),
            "W": self.W.copy(),
            "b": np.asarray(self.b),
            "mV": self.mV.copy(),
            "vV": self.vV.copy(),
            "mW": self.mW.copy(),
            "vW": self.vW.copy(),
            "t": np.asarray(self.t, dtype=np.int64),
        }

    def load_optimizer_state(self, state: dict) -> None:
        for key in ("V", "W", "mV", "vV", "mW", "vW"):
            setattr(self, key, np.asarray(state[key]).copy())
        self.b = np.float32(np.asarray(state["b"]).item())
        self.t = int(np.asarray(state["t"]).item())
