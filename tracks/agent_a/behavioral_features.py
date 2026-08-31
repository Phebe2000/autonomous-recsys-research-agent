"""Leakage-safe train-causal behavioral features for logged exposures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from data import SPLITS

from .fingerprint import canonical_json
from .safe_data import STANDARD_LOGS, TrainFittedEncoder, UnlabeledExposure


FEATURE_SCHEMA_VERSION = 1
DEFAULT_SMOOTHING = 20.0


@dataclass(frozen=True)
class ExposureContext:
    row_index: int
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    hourmin: int
    time_ms: int
    label: int | None


@dataclass(frozen=True)
class BehavioralFeatureBundle:
    train: np.ndarray
    evaluation: np.ndarray
    train_users: tuple[str, ...]
    evaluation_users: tuple[str, ...]
    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    schema_sha256: str


def _authors(data_dir: Path) -> dict[str, str]:
    with (Path(data_dir) / "video_features_basic_pure.csv").open(newline="") as stream:
        return {row["video_id"]: row["author_id"] for row in csv.DictReader(stream)}


def _contexts(data_dir: Path, split: str) -> list[ExposureContext]:
    if split not in {"train", "valid", "test"}:
        raise ValueError("behavioral features require train, valid, or test")
    authors = _authors(data_dir)
    low, high = SPLITS[split]
    output = []
    for filename in STANDARD_LOGS:
        with (Path(data_dir) / filename).open(newline="") as stream:
            for raw in csv.DictReader(stream):
                date = int(raw["date"])
                if not low <= date <= high:
                    continue
                # Feature construction indexes relevance only for train rows.
                label = int(raw["long_view"] != "0") if split == "train" else None
                output.append(ExposureContext(
                    len(output), date, raw["user_id"], raw["video_id"],
                    authors.get(raw["video_id"], "UNK"), raw["tab"],
                    float(raw["duration_ms"]), int(raw["hourmin"]),
                    int(raw["time_ms"]), label,
                ))
    return output


def _duration_edges(train: list[ExposureContext]) -> np.ndarray:
    values = np.asarray([row.duration_ms for row in train], dtype=np.float64)
    return np.quantile(values, np.linspace(0.0, 1.0, 11)[1:-1])


def _entity_keys(encoded_row: np.ndarray) -> tuple[Any, ...]:
    user, video, author, _, duration_bucket = (int(value) for value in encoded_row)
    return (
        video,
        author,
        (user << 32) | video,
        (user << 32) | author,
        (user << 32) | duration_bucket,
    )


def _rate_features(
    keys: tuple[Any, ...],
    tables: list[dict[Any, tuple[int, int]]],
    prior: float,
    smoothing: float,
) -> list[float]:
    output = []
    for key, table in zip(keys, tables):
        count, positives = table.get(key, (0, 0))
        output.extend((np.log1p(count), (positives + smoothing * prior) / (count + smoothing)))
    return output


def _direct_features(row: ExposureContext, train_start: int) -> list[float]:
    minute = (row.hourmin // 100) * 60 + row.hourmin % 100
    phase = 2.0 * np.pi * minute / 1440.0
    return [
        np.log1p(max(row.duration_ms, 0.0)),
        np.sin(phase),
        np.cos(phase),
        float(row.date - train_start),
    ]


def _validate_alignment(
    contexts: list[ExposureContext],
    encoded: np.ndarray,
    expected_users: list[str] | tuple[str, ...],
) -> None:
    if len(contexts) != len(encoded) or len(contexts) != len(expected_users):
        raise ValueError("behavioral feature rows are not aligned with encoded exposures")
    for index, (row, user) in enumerate(zip(contexts, expected_users)):
        if row.row_index != index or row.user_id != str(user):
            raise ValueError(f"behavioral feature identity mismatch at row {index}")


def build_behavioral_features(
    data_dir: Path,
    enc: dict,
    encoder: TrainFittedEncoder,
    evaluation_split: str = "valid",
    smoothing: float = DEFAULT_SMOOTHING,
) -> BehavioralFeatureBundle:
    """Build causal training aggregates and frozen train-only eval features."""
    if evaluation_split not in {"valid", "test"}:
        raise ValueError("evaluation split must be valid or test")
    if smoothing <= 0:
        raise ValueError("behavioral smoothing must be positive")
    train = _contexts(Path(data_dir), "train")
    evaluation = _contexts(Path(data_dir), evaluation_split)
    eval_encoded = enc[evaluation_split][0]
    eval_users = enc[evaluation_split][2]
    _validate_alignment(train, enc["train"][0], enc["train"][2])
    _validate_alignment(evaluation, eval_encoded, eval_users)
    tables: list[dict[Any, tuple[int, int]]] = [dict() for _ in range(5)]
    train_extra = np.empty((len(train), 14), dtype=np.float64)
    positives = 0
    seen = 0
    order = sorted(range(len(train)), key=lambda index: (train[index].time_ms, index))
    cursor = 0
    while cursor < len(order):
        timestamp = train[order[cursor]].time_ms
        stop = cursor
        while stop < len(order) and train[order[stop]].time_ms == timestamp:
            stop += 1
        prior = 0.5 if seen == 0 else positives / seen
        for index in order[cursor:stop]:
            row = train[index]
            keys = _entity_keys(enc["train"][0][index] - encoder.offsets)
            train_extra[index] = _direct_features(row, SPLITS["train"][0]) + _rate_features(
                keys, tables, prior, smoothing
            )
        # Tied timestamps update only after every row at that timestamp was featurized.
        for index in order[cursor:stop]:
            row = train[index]
            label = int(row.label)
            for key, table in zip(
                _entity_keys(enc["train"][0][index] - encoder.offsets), tables
            ):
                count, total = table.get(key, (0, 0))
                table[key] = (count + 1, total + label)
            positives += label
            seen += 1
        cursor = stop
    frozen_prior = positives / max(seen, 1)
    eval_extra = np.empty((len(evaluation), 14), dtype=np.float64)
    for index, row in enumerate(evaluation):
        eval_extra[index] = _direct_features(row, SPLITS["train"][0]) + _rate_features(
            _entity_keys(eval_encoded[index] - encoder.offsets), tables, frozen_prior, smoothing
        )
    train_categorical = np.asarray(enc["train"][0], dtype=np.float64).copy()
    eval_categorical = np.asarray(eval_encoded, dtype=np.float64).copy()
    for column, offset in enumerate(encoder.offsets):
        train_categorical[:, column] -= offset
        eval_categorical[:, column] -= offset
    train_matrix = np.column_stack((train_categorical, train_extra))
    eval_matrix = np.column_stack((eval_categorical, eval_extra))
    if not np.all(np.isfinite(train_matrix)) or not np.all(np.isfinite(eval_matrix)):
        raise ValueError("behavioral features must be finite")
    names = (
        "user_id", "video_id", "author_id", "tab", "duration_bucket",
        "log_duration", "hour_sin", "hour_cos", "days_from_train_start",
        "video_count", "video_long_view_rate",
        "author_count", "author_long_view_rate",
        "user_video_count", "user_video_long_view_rate",
        "user_author_count", "user_author_long_view_rate",
        "user_duration_count", "user_duration_long_view_rate",
    )
    schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": names,
        "categorical_indices": list(range(5)),
        "smoothing": smoothing,
        "train_aggregate_policy": "strictly_earlier_time_ms_excluding_ties",
        "evaluation_aggregate_policy": "frozen_train_only",
    }
    digest = hashlib.sha256(canonical_json(schema).encode()).hexdigest()
    return BehavioralFeatureBundle(
        train_matrix, eval_matrix,
        tuple(row.user_id for row in train),
        tuple(row.user_id for row in evaluation),
        names, tuple(range(5)), digest,
    )
