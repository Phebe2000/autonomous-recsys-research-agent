"""Training-only raw auxiliary-label adapter and stable objective primitives."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data import SPLITS


TRAIN_LOG = "log_standard_4_08_to_4_21_pure.csv"


@dataclass(frozen=True)
class RawIdentity:
    date: int
    user_id: str
    video_id: str
    time_ms: int


@dataclass(frozen=True)
class PlayTimeTransform:
    mean: float
    scale: float

    @classmethod
    def fit_train(cls, play_time_ms: Sequence[float]) -> "PlayTimeTransform":
        values = np.asarray(play_time_ms, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("training play_time_ms must be a non-empty finite non-negative vector")
        logged = np.log1p(values)
        scale = float(logged.std())
        return cls(float(logged.mean()), scale if scale > 0 else 1.0)

    def transform(self, play_time_ms: Sequence[float]) -> np.ndarray:
        values = np.asarray(play_time_ms, dtype=np.float64)
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("play_time_ms must be finite and non-negative")
        return ((np.log1p(values) - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "scale": self.scale}


class AuxiliaryTrainLabels:
    def __init__(
        self,
        identities: tuple[RawIdentity, ...],
        is_click: np.ndarray,
        play_time_ms: np.ndarray,
        play_transform: PlayTimeTransform,
    ) -> None:
        self.identities = identities
        self.is_click = np.asarray(is_click, dtype=np.float32)
        self.play_time_ms = np.asarray(play_time_ms, dtype=np.float32)
        self.play_transform = play_transform
        self.play_target = play_transform.transform(self.play_time_ms)
        lengths = {len(identities), len(self.is_click), len(self.play_time_ms), len(self.play_target)}
        if len(lengths) != 1:
            raise ValueError("auxiliary training arrays must align")

    def for_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split != "train":
            raise ValueError("auxiliary labels are available only for train")
        return self.is_click, self.play_target


def _raw_train_rows(data_dir: Path):
    low, high = SPLITS["train"]
    with (Path(data_dir) / TRAIN_LOG).open(newline="") as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            if low <= date <= high:
                yield row


def read_training_identities(data_dir: Path) -> tuple[RawIdentity, ...]:
    return tuple(
        RawIdentity(int(row["date"]), row["user_id"], row["video_id"], int(row["time_ms"]))
        for row in _raw_train_rows(data_dir)
    )


def load_training_auxiliary(
    data_dir: Path,
    splits: dict,
    enc: dict,
    expected_identities: Sequence[RawIdentity] | None = None,
) -> AuxiliaryTrainLabels:
    """Load train labels and fail on any raw/official/encoded row mismatch."""
    rows = list(_raw_train_rows(data_dir))
    official = splits["train"]
    encoded_x, encoded_y, encoded_users = enc["train"]
    if len(rows) != len(official) or len(encoded_x) != len(official):
        raise ValueError("auxiliary raw/official/encoded train row count mismatch")
    if len(encoded_y) != len(official) or len(encoded_users) != len(official):
        raise ValueError("encoded train arrays are not row-aligned")
    if expected_identities is not None and len(expected_identities) != len(rows):
        raise ValueError("expected auxiliary identity row count mismatch")
    identities = []
    clicks = np.empty(len(rows), dtype=np.float32)
    play = np.empty(len(rows), dtype=np.float32)
    for index, (raw, official_row, encoded_user) in enumerate(zip(rows, official, encoded_users)):
        identity = RawIdentity(
            int(raw["date"]), raw["user_id"], raw["video_id"], int(raw["time_ms"])
        )
        if expected_identities is not None and identity != expected_identities[index]:
            raise ValueError(f"auxiliary identity mismatch at train row {index}")
        if (identity.date, identity.user_id, identity.video_id) != (
            int(official_row[0]), str(official_row[1]), str(official_row[2])
        ):
            raise ValueError(f"auxiliary official-row mismatch at train row {index}")
        if identity.user_id != str(encoded_user):
            raise ValueError(f"auxiliary encoded-user mismatch at train row {index}")
        long_view = 1.0 if raw["long_view"] != "0" else 0.0
        if long_view != float(encoded_y[index]):
            raise ValueError(f"auxiliary encoded-label mismatch at train row {index}")
        click = float(raw["is_click"])
        play_time = float(raw["play_time_ms"])
        if click not in (0.0, 1.0):
            raise ValueError(f"invalid is_click at train row {index}")
        if play_time < 0 or not np.isfinite(play_time):
            raise ValueError(f"invalid play_time_ms at train row {index}")
        identities.append(identity)
        clicks[index] = click
        play[index] = play_time
    transform = PlayTimeTransform.fit_train(play)
    return AuxiliaryTrainLabels(tuple(identities), clicks, play, transform)


def binary_cross_entropy_with_gradient(logits, targets) -> tuple[float, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if logits.shape != targets.shape or logits.ndim != 1 or np.any((targets < 0) | (targets > 1)):
        raise ValueError("BCE logits and binary targets must be aligned vectors")
    loss = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    gradient = ((1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700))) - targets) / len(logits))
    result = float(loss.mean())
    if not np.isfinite(result) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite BCE")
    return result, gradient


def huber_with_gradient(predictions, targets, delta: float = 1.0) -> tuple[float, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 1 or delta <= 0:
        raise ValueError("Huber predictions/targets must align and delta must be positive")
    error = predictions - targets
    absolute = np.abs(error)
    quadratic = absolute <= delta
    losses = np.where(quadratic, 0.5 * error * error, delta * (absolute - 0.5 * delta))
    gradient = np.where(quadratic, error, delta * np.sign(error)) / len(error)
    result = float(losses.mean())
    if not np.isfinite(result) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("non-finite Huber")
    return result, gradient
