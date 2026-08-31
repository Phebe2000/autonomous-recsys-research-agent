"""FM + causal positive-history mean-pooling residual for Agent A."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from .fingerprint import canonical_json
from .history import CausalPositiveHistory, csr_mean_pool
from .listwise import listnet_loss_and_gradient


def history_residual_and_gradients(
    candidate_ids: np.ndarray,
    indptr: np.ndarray,
    history_ids: np.ndarray,
    embeddings: np.ndarray,
    gate: float,
    upstream: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Compute residual and, when upstream is supplied, H/gate gradients."""
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    indptr = np.asarray(indptr, dtype=np.int64)
    history_ids = np.asarray(history_ids, dtype=np.int64)
    if len(indptr) != len(candidate_ids) + 1:
        raise ValueError("one CSR history row is required per candidate")
    if len(candidate_ids) and (
        np.any(candidate_ids < 0) or np.any(candidate_ids >= len(embeddings))
    ):
        raise IndexError("candidate id is outside history embedding table")
    means = csr_mean_pool(indptr, history_ids, embeddings)
    scale = float(np.sqrt(embeddings.shape[1]))
    similarity = np.sum(embeddings[candidate_ids] * means, axis=1) / scale
    residual = float(gate) * similarity
    if upstream is None:
        return residual, None, None
    upstream = np.asarray(upstream, dtype=np.float64)
    if len(upstream) != len(candidate_ids):
        raise ValueError("upstream gradient must align with candidate rows")
    grad_h = np.zeros_like(embeddings, dtype=np.float64)
    candidate_vectors = embeddings[candidate_ids].astype(np.float64)
    np.add.at(
        grad_h,
        candidate_ids,
        upstream[:, None] * float(gate) * means.astype(np.float64) / scale,
    )
    lengths = np.diff(indptr)
    if len(history_ids):
        owners = np.repeat(np.arange(len(candidate_ids)), lengths)
        contribution = (
            upstream[owners, None]
            * float(gate)
            * candidate_vectors[owners]
            / (scale * lengths[owners, None])
        )
        np.add.at(grad_h, history_ids, contribution)
    grad_gate = float(np.dot(upstream, similarity))
    return residual, grad_h, grad_gate


class HistoryListwiseFM:
    """Listwise FM with an independently optimized history embedding table."""

    def __init__(
        self,
        baseline,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        history_dim: int = 16,
        gate: float = 1.0,
        history_lr: float | None = None,
        gate_lr: float | None = None,
        history_unfreeze_step: int = 0,
        train_history_embeddings: bool = True,
        train_gate: bool = True,
    ) -> None:
        self.V = baseline.V.copy()
        self.W = baseline.W.copy()
        self.b = np.float32(baseline.b)
        if history_dim != self.V.shape[1]:
            raise ValueError("controlled history runs require history_dim equal to baseline k")
        # Independent storage initialized from this dataset's fresh baseline only.
        self.H = baseline.V.copy()
        self.gate = np.float32(gate)
        self.lr = float(lr)
        self.history_lr = None if history_lr is None else float(history_lr)
        self.gate_lr = None if gate_lr is None else float(gate_lr)
        self.history_unfreeze_step = int(history_unfreeze_step)
        self.train_history_embeddings = bool(train_history_embeddings)
        self.train_gate = bool(train_gate)
        if self.history_unfreeze_step < 0:
            raise ValueError("history_unfreeze_step must be non-negative")
        self.weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mH = np.zeros_like(self.H)
        self.vH = np.zeros_like(self.H)
        self.mg = np.float32(0.0)
        self.vg = np.float32(0.0)
        self.t = 0
        self.tH = 0

    def fm_forward(self, X: np.ndarray):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * ((summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2)))
        return self.b + self.W[X].sum(axis=1) + interactions, embeddings, summed

    def scores(self, X: np.ndarray, indptr: np.ndarray, history_ids: np.ndarray) -> np.ndarray:
        if len(indptr) != len(X) + 1:
            raise ValueError("history rows and candidates must have identical lengths")
        fm_scores = self.fm_forward(X)[0]
        if float(self.gate) == 0.0:
            return fm_scores
        residual = history_residual_and_gradients(
            X[:, 1], indptr, history_ids, self.H, float(self.gate)
        )[0]
        if len(residual) != len(fm_scores):
            raise AssertionError("history residual length mismatch")
        return fm_scores + residual.astype(fm_scores.dtype, copy=False)

    def step(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        group_sizes: Sequence[int],
        indptr: np.ndarray,
        history_ids: np.ndarray,
        score_temperature: float,
        target_temperature: float,
        group_weights: Sequence[float] | None,
        update_embeddings: bool = True,
    ) -> float:
        if len(X) != len(labels) or len(indptr) != len(X) + 1:
            raise ValueError("features, labels, and history rows must align")
        fm_scores, fm_embeddings, summed = self.fm_forward(X)
        residual, _, _ = history_residual_and_gradients(
            X[:, 1], indptr, history_ids, self.H, float(self.gate)
        )
        scores = fm_scores + residual.astype(fm_scores.dtype, copy=False)
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
            score_gradient[:, None, None] * (summed[:, None, :] - fm_embeddings),
        )
        _, grad_h, grad_gate = history_residual_and_gradients(
            X[:, 1], indptr, history_ids, self.H, float(self.gate), score_gradient
        )
        grad_h = grad_h.astype(np.float32)
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
        if self.train_history_embeddings and self.t > self.history_unfreeze_step:
            self.tH += 1
            history_rate = rate if self.history_lr is None else self.history_lr
            self.mH *= beta1
            self.mH += (1 - beta1) * grad_h
            self.vH *= beta2
            self.vH += (1 - beta2) * grad_h * grad_h
            self.H *= 1.0 - history_rate * self.weight_decay
            self.H -= history_rate * (self.mH / (1 - beta1**self.tH)) / (
                np.sqrt(self.vH / (1 - beta2**self.tH)) + epsilon
            )
        if self.train_gate:
            self.mg = np.float32(beta1 * self.mg + (1 - beta1) * grad_gate)
            self.vg = np.float32(beta2 * self.vg + (1 - beta2) * grad_gate * grad_gate)
            gate_rate = rate if self.gate_lr is None else self.gate_lr
            self.gate -= np.float32(
                gate_rate
                * (self.mg / (1 - beta1**self.t))
                / (np.sqrt(self.vg / (1 - beta2**self.t)) + epsilon)
            )
        return float(loss)

    def predict(
        self,
        X: np.ndarray,
        history: CausalPositiveHistory,
        split: str,
        limit: int | None,
        batch_size: int = 100_000,
    ) -> np.ndarray:
        outputs = []
        for start in range(0, len(X), batch_size):
            stop = min(len(X), start + batch_size)
            rows = np.arange(start, stop, dtype=np.int64)
            indptr, ids = history.batch_csr(split, rows, limit)
            outputs.append(self.scores(X[start:stop], indptr, ids))
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def snapshot(self) -> dict[str, Any]:
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in {
                "V": self.V,
                "W": self.W,
                "b": np.asarray(self.b),
                "H": self.H,
                "gate": np.asarray(self.gate),
                "mV": self.mV,
                "vV": self.vV,
                "mW": self.mW,
                "vW": self.vW,
                "mH": self.mH,
                "vH": self.vH,
                "mg": np.asarray(self.mg),
                "vg": np.asarray(self.vg),
                "t": np.asarray(self.t, dtype=np.int64),
                "tH": np.asarray(self.tH, dtype=np.int64),
            }.items()
        }

    def load_snapshot(self, state: dict[str, Any]) -> None:
        for key in ("V", "W", "H", "mV", "vV", "mW", "vW", "mH", "vH"):
            setattr(self, key, np.asarray(state[key]).copy())
        for key in ("b", "gate", "mg", "vg"):
            setattr(self, key, np.float32(np.asarray(state[key]).item()))
        self.t = int(np.asarray(state["t"]).item())
        self.tH = int(np.asarray(state["tH"]).item())


def save_history_checkpoint(
    path: Path,
    model: HistoryListwiseFM,
    dataset_fingerprint: str,
    trial_id: str,
    config: dict,
    progress: dict,
) -> dict:
    """Atomically save parameters, optimizer state, config, and resume metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = canonical_json(config)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    payload = {
        **model.snapshot(),
        "dataset_fingerprint": np.asarray(dataset_fingerprint),
        "trial_id": np.asarray(trial_id),
        "config_json": np.asarray(config_json),
        "config_hash": np.asarray(config_hash),
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
    return {"config_hash": config_hash, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_history_checkpoint(
    path: Path,
    model: HistoryListwiseFM,
    expected_fingerprint: str,
    expected_trial_id: str,
    expected_config: dict,
) -> dict:
    expected_json = canonical_json(expected_config)
    expected_hash = hashlib.sha256(expected_json.encode()).hexdigest()
    with np.load(Path(path), allow_pickle=False) as saved:
        if str(saved["dataset_fingerprint"]) != expected_fingerprint:
            raise ValueError("history checkpoint dataset fingerprint mismatch")
        if str(saved["trial_id"]) != expected_trial_id:
            raise ValueError("history checkpoint trial id mismatch")
        if str(saved["config_hash"]) != expected_hash or str(saved["config_json"]) != expected_json:
            raise ValueError("history checkpoint config mismatch")
        state = {key: saved[key].copy() for key in model.snapshot()}
        progress = json.loads(str(saved["progress_json"]))
    model.load_snapshot(state)
    return progress
