"""Judged-run data boundary that never exposes hidden-test labels.

The official starter-kit files remain unchanged. Research code receives labeled
train/validation rows only; final scoring receives unlabeled exposure rows.
Feature fitting is train-only and reproduces the five official FM fields.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from data import FIELDS, SPLITS


STANDARD_LOGS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)
VIDEO_BASIC = "video_features_basic_pure.csv"


@dataclass(frozen=True)
class UnlabeledExposure:
    date: int
    user_id: str
    video_id: str
    author_id: str
    tab: str
    duration_ms: float
    hourmin: int
    time_ms: int

    def submission_row(self) -> tuple:
        # Compatible with official submit.write_submission, which reads [1]/[2].
        return (self.date, self.user_id, self.video_id, self.author_id, self.tab, self.duration_ms)


def _authors(data_dir: Path) -> dict[str, str]:
    with (Path(data_dir) / VIDEO_BASIC).open(newline="") as stream:
        return {row["video_id"]: row["author_id"] for row in csv.DictReader(stream)}


def _standard_rows(data_dir: Path) -> Iterable[dict[str, str]]:
    for filename in STANDARD_LOGS:
        with (Path(data_dir) / filename).open(newline="") as stream:
            yield from csv.DictReader(stream)


def load_research_splits(data_dir: Path) -> dict[str, list[tuple]]:
    """Load labels only for train and public validation dates.

    Rows in the hidden-test date range are skipped before ``long_view`` is ever
    indexed. The returned mapping has no ``test`` key by construction.
    """
    data_dir = Path(data_dir)
    authors = _authors(data_dir)
    output: dict[str, list[tuple]] = {"train": [], "valid": []}
    for raw in _standard_rows(data_dir):
        date = int(raw["date"])
        split = next(
            (
                name
                for name in ("train", "valid")
                if SPLITS[name][0] <= date <= SPLITS[name][1]
            ),
            None,
        )
        if split is None:
            continue
        output[split].append(
            (
                date,
                raw["user_id"],
                raw["video_id"],
                authors.get(raw["video_id"], "UNK"),
                raw["tab"],
                float(raw["duration_ms"]),
                1 if raw["long_view"] != "0" else 0,
            )
        )
    return output


def load_unlabeled_exposures(data_dir: Path, split: str) -> list[UnlabeledExposure]:
    """Load inference fields without indexing the relevance-label column."""
    if split not in {"valid", "test"}:
        raise ValueError("unlabeled exposures are available only for valid/test")
    data_dir = Path(data_dir)
    authors = _authors(data_dir)
    low, high = SPLITS[split]
    output = []
    for raw in _standard_rows(data_dir):
        date = int(raw["date"])
        if low <= date <= high:
            output.append(
                UnlabeledExposure(
                    date,
                    raw["user_id"],
                    raw["video_id"],
                    authors.get(raw["video_id"], "UNK"),
                    raw["tab"],
                    float(raw["duration_ms"]),
                    int(raw["hourmin"]),
                    int(raw["time_ms"]),
                )
            )
    return output


@dataclass(frozen=True)
class TrainFittedEncoder:
    edges: np.ndarray
    vocabs: tuple[dict[str, int], ...]
    unknown: tuple[int, ...]
    offsets: np.ndarray
    dimension: int

    @classmethod
    def fit(cls, train_rows: Sequence[tuple]) -> "TrainFittedEncoder":
        if not train_rows:
            raise ValueError("cannot fit encoder without training rows")
        durations = np.asarray([row[5] for row in train_rows], dtype=np.float64)
        edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
        vocabs: list[dict[str, int]] = [dict() for _ in FIELDS]
        for row in train_rows:
            for index, value in enumerate(_raw_features(row, edges)):
                if value not in vocabs[index]:
                    vocabs[index][value] = len(vocabs[index])
        unknown = tuple(len(vocab) for vocab in vocabs)
        field_dims = [len(vocab) + 1 for vocab in vocabs]
        offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
        return cls(edges, tuple(vocabs), unknown, offsets, int(sum(field_dims)))

    def transform(self, rows: Sequence[tuple] | Sequence[UnlabeledExposure]) -> np.ndarray:
        matrix = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for row_index, row in enumerate(rows):
            for field_index, value in enumerate(_raw_features(row, self.edges)):
                matrix[row_index, field_index] = (
                    self.vocabs[field_index].get(value, self.unknown[field_index])
                    + self.offsets[field_index]
                )
        return matrix


def _raw_features(row: tuple | UnlabeledExposure, edges: np.ndarray) -> list[str]:
    if isinstance(row, UnlabeledExposure):
        user, video, author, tab, duration = (
            row.user_id, row.video_id, row.author_id, row.tab, row.duration_ms
        )
    else:
        user, video, author, tab, duration = row[1], row[2], row[3], row[4], row[5]
    return [
        str(user),
        str(video),
        str(author),
        str(tab),
        str(int(np.searchsorted(edges, float(duration)))),
    ]


def encode_research_splits(
    splits: dict[str, list[tuple]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, list[str]]], TrainFittedEncoder]:
    if set(splits) != {"train", "valid"}:
        raise ValueError("research encoder accepts exactly train and valid splits")
    encoder = TrainFittedEncoder.fit(splits["train"])
    encoded = {}
    for name in ("train", "valid"):
        rows = splits[name]
        encoded[name] = (
            encoder.transform(rows),
            np.asarray([row[6] for row in rows], dtype=np.float32),
            [str(row[1]) for row in rows],
        )
    return encoded, encoder


def load_safe_side_features(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    """Return numeric feature views with feedback exposed for train only."""
    if split not in {"train", "valid", "test"}:
        raise ValueError("unknown side-feature split")
    low, high = SPLITS[split]
    common: dict[str, list[float]] = {
        "date": [], "hourmin": [], "time_ms": [], "log_duration_ms": [],
    }
    train_only: dict[str, list[float]] = {
        "is_click": [], "log_play_time_ms": [],
    }
    for raw in _standard_rows(Path(data_dir)):
        date = int(raw["date"])
        if not low <= date <= high:
            continue
        common["date"].append(float(date))
        common["hourmin"].append(float(raw["hourmin"]))
        common["time_ms"].append(float(raw["time_ms"]))
        common["log_duration_ms"].append(float(np.log1p(float(raw["duration_ms"]))))
        if split == "train":
            train_only["is_click"].append(float(raw["is_click"]))
            train_only["log_play_time_ms"].append(float(np.log1p(float(raw["play_time_ms"]))))
    output = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in common.items()
    }
    if split == "train":
        output.update({
            key: np.asarray(values, dtype=np.float64)
            for key, values in train_only.items()
        })
    lengths = {len(values) for values in output.values()}
    if len(lengths) != 1:
        raise ValueError("safe side-feature arrays are not row-aligned")
    return output
