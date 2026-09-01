"""Shared-FM multi-task model with train-only auxiliary heads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from .auxiliary import binary_cross_entropy_with_gradient, huber_with_gradient
from .fingerprint import canonical_json
from .listwise import ListwiseFM, listnet_loss_and_gradient


class MultiTaskListwiseFM:
    def __init__(
        self,
        baseline,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        click_weight: float,
        play_weight: float,
        head_lr: float | None = None,
        huber_delta: float = 1.0,
    ) -> None:
        if click_weight < 0 or play_weight < 0:
            raise ValueError("auxiliary weights must be non-negative")
        self.core = ListwiseFM(baseline, lr, weight_decay, warmup_steps)
        k = self.core.V.shape[1]
        self.click_head = np.zeros(k, dtype=np.float32)
        self.play_head = np.zeros(k, dtype=np.float32)
        self.click_bias = np.float32(0.0)
        self.play_bias = np.float32(0.0)
        self.m_click = np.zeros(k, dtype=np.float32)
        self.v_click = np.zeros(k, dtype=np.float32)
        self.m_play = np.zeros(k, dtype=np.float32)
        self.v_play = np.zeros(k, dtype=np.float32)
        self.m_click_bias = np.float32(0.0)
        self.v_click_bias = np.float32(0.0)
        self.m_play_bias = np.float32(0.0)
        self.v_play_bias = np.float32(0.0)
        self.click_weight = float(click_weight)
        self.play_weight = float(play_weight)
        self.head_lr = float(lr if head_lr is None else head_lr)
        self.huber_delta = float(huber_delta)

    @property
    def V(self):
        return self.core.V

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        result = self.core.predict(X, batch_size)
        if len(result) != len(X):
            raise AssertionError("one long_view score is required per exposure row")
        return result

    @staticmethod
    def _adam(parameter, gradient, first, second, rate, t, decay=0.0):
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        first *= beta1
        first += (1 - beta1) * gradient
        second *= beta2
        second += (1 - beta2) * gradient * gradient
        parameter *= 1.0 - rate * decay
        parameter -= rate * (first / (1 - beta1**t)) / (
            np.sqrt(second / (1 - beta2**t)) + epsilon
        )

    def step(
        self,
        X: np.ndarray,
        long_view: np.ndarray,
        click: np.ndarray,
        play_target: np.ndarray,
        group_sizes: Sequence[int],
        score_temperature: float,
        target_temperature: float,
        group_weights: Sequence[float] | None,
        update_embeddings: bool = True,
    ) -> dict[str, float]:
        lengths = {len(X), len(long_view), len(click), len(play_target)}
        if len(lengths) != 1:
            raise ValueError("long_view and auxiliary labels must align with exposure rows")
        if self.click_weight == 0.0 and self.play_weight == 0.0:
            loss = self.core.step(
                X, long_view, group_sizes, score_temperature, target_temperature,
                group_weights, update_embeddings,
            )
            return {"total": loss, "listnet": loss, "click": 0.0, "play": 0.0}

        scores, embeddings, summed = self.core.logits(X)
        listnet_loss, score_gradient = listnet_loss_and_gradient(
            scores, long_view, group_sizes, score_temperature, target_temperature, group_weights
        )
        click_logits = summed @ self.click_head + self.click_bias
        play_predictions = summed @ self.play_head + self.play_bias
        click_loss, click_gradient = binary_cross_entropy_with_gradient(click_logits, click)
        play_loss, play_gradient = huber_with_gradient(
            play_predictions, play_target, self.huber_delta
        )
        score_gradient = score_gradient.astype(np.float32)
        click_gradient = click_gradient.astype(np.float32) * self.click_weight
        play_gradient = play_gradient.astype(np.float32) * self.play_weight

        grad_w = np.zeros_like(self.core.W)
        grad_v = np.zeros_like(self.core.V)
        np.add.at(grad_w, X, score_gradient[:, None])
        np.add.at(
            grad_v, X,
            score_gradient[:, None, None] * (summed[:, None, :] - embeddings),
        )
        auxiliary_sum_gradient = (
            click_gradient[:, None] * self.click_head[None, :]
            + play_gradient[:, None] * self.play_head[None, :]
        )
        np.add.at(grad_v, X, auxiliary_sum_gradient[:, None, :])
        grad_click_head = summed.T @ click_gradient
        grad_play_head = summed.T @ play_gradient
        grad_click_bias = np.float32(click_gradient.sum())
        grad_play_bias = np.float32(play_gradient.sum())

        self.core.t += 1
        t = self.core.t
        rate = self.core.lr * (
            min(1.0, t / self.core.warmup_steps) if self.core.warmup_steps else 1.0
        )
        if update_embeddings:
            self._adam(
                self.core.V, grad_v, self.core.mV, self.core.vV,
                rate, t, self.core.weight_decay,
            )
        self._adam(
            self.core.W, grad_w, self.core.mW, self.core.vW,
            rate, t, self.core.weight_decay,
        )
        self._adam(self.click_head, grad_click_head, self.m_click, self.v_click, self.head_lr, t)
        self._adam(self.play_head, grad_play_head, self.m_play, self.v_play, self.head_lr, t)
        for name, gradient in (("click", grad_click_bias), ("play", grad_play_bias)):
            first_name, second_name, bias_name = f"m_{name}_bias", f"v_{name}_bias", f"{name}_bias"
            first = np.float32(0.9 * getattr(self, first_name) + 0.1 * gradient)
            second = np.float32(0.999 * getattr(self, second_name) + 0.001 * gradient * gradient)
            bias = getattr(self, bias_name) - np.float32(
                self.head_lr * (first / (1 - 0.9**t)) /
                (np.sqrt(second / (1 - 0.999**t)) + 1e-8)
            )
            setattr(self, first_name, first)
            setattr(self, second_name, second)
            setattr(self, bias_name, np.float32(bias))
        return {
            "total": float(listnet_loss + self.click_weight * click_loss + self.play_weight * play_loss),
            "listnet": float(listnet_loss),
            "click": float(click_loss),
            "play": float(play_loss),
        }

    def snapshot(self) -> dict[str, Any]:
        values = {
            "V": self.core.V, "W": self.core.W, "b": np.asarray(self.core.b),
            "mV": self.core.mV, "vV": self.core.vV,
            "mW": self.core.mW, "vW": self.core.vW,
            "t": np.asarray(self.core.t, dtype=np.int64),
            "click_head": self.click_head, "play_head": self.play_head,
            "click_bias": np.asarray(self.click_bias), "play_bias": np.asarray(self.play_bias),
            "m_click": self.m_click, "v_click": self.v_click,
            "m_play": self.m_play, "v_play": self.v_play,
            "m_click_bias": np.asarray(self.m_click_bias),
            "v_click_bias": np.asarray(self.v_click_bias),
            "m_play_bias": np.asarray(self.m_play_bias),
            "v_play_bias": np.asarray(self.v_play_bias),
        }
        return {key: value.copy() if hasattr(value, "copy") else value for key, value in values.items()}

    def load_snapshot(self, state: dict[str, Any]) -> None:
        for key in ("V", "W", "mV", "vV", "mW", "vW"):
            setattr(self.core, key, np.asarray(state[key]).copy())
        self.core.b = np.float32(np.asarray(state["b"]).item())
        self.core.t = int(np.asarray(state["t"]).item())
        for key in ("click_head", "play_head", "m_click", "v_click", "m_play", "v_play"):
            setattr(self, key, np.asarray(state[key]).copy())
        for key in (
            "click_bias", "play_bias", "m_click_bias", "v_click_bias",
            "m_play_bias", "v_play_bias",
        ):
            setattr(self, key, np.float32(np.asarray(state[key]).item()))


def save_multitask_checkpoint(
    path: Path,
    model: MultiTaskListwiseFM,
    dataset_fingerprint: str,
    trial_id: str,
    config: dict,
    normalization: dict,
    progress: dict,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = canonical_json(config)
    payload = {
        **model.snapshot(),
        "dataset_fingerprint": np.asarray(dataset_fingerprint),
        "trial_id": np.asarray(trial_id),
        "config_json": np.asarray(config_json),
        "config_hash": np.asarray(hashlib.sha256(config_json.encode()).hexdigest()),
        "normalization_json": np.asarray(canonical_json(normalization)),
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
    return {"config_hash": str(payload["config_hash"])}


def load_multitask_checkpoint(
    path: Path,
    model: MultiTaskListwiseFM,
    dataset_fingerprint: str,
    trial_id: str,
    config: dict,
    normalization: dict,
) -> dict:
    expected_config = canonical_json(config)
    expected_normalization = canonical_json(normalization)
    with np.load(path, allow_pickle=False) as checkpoint:
        if str(checkpoint["dataset_fingerprint"]) != dataset_fingerprint:
            raise ValueError("multi-task checkpoint fingerprint mismatch")
        if str(checkpoint["trial_id"]) != trial_id:
            raise ValueError("multi-task checkpoint trial mismatch")
        if str(checkpoint["config_json"]) != expected_config:
            raise ValueError("multi-task checkpoint config mismatch")
        if str(checkpoint["normalization_json"]) != expected_normalization:
            raise ValueError("multi-task checkpoint normalization mismatch")
        model.load_snapshot({key: checkpoint[key] for key in model.snapshot()})
        return json.loads(str(checkpoint["progress_json"]))
