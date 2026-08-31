"""Strictly causal positive-history sidecar for Agent A.

This is a mean-pooling input builder, not an order-preserving sequence encoder.
Training membership is row-specific and strictly earlier by ``time_ms``.
Validation membership is a frozen snapshot of train positives only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data import SPLITS


@dataclass(frozen=True)
class RawSidecarRow:
    date: int
    user_id: str
    video_id: str
    time_ms: int


class CausalPositiveHistory:
    def __init__(
        self,
        positive_times: tuple[np.ndarray, ...],
        positive_items: tuple[np.ndarray, ...],
        train_user_slots: np.ndarray,
        train_cutoffs: np.ndarray,
        valid_user_slots: np.ndarray,
    ) -> None:
        self.positive_times = positive_times
        self.positive_items = positive_items
        self.train_user_slots = train_user_slots.astype(np.int32, copy=False)
        self.train_cutoffs = train_cutoffs.astype(np.int32, copy=False)
        self.valid_user_slots = valid_user_slots.astype(np.int32, copy=False)

    def batch_csr(
        self, split: str, row_indices: Sequence[int] | np.ndarray, limit: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialize last-N history for requested rows as CSR."""
        if split not in {"train", "valid"}:
            raise ValueError("history is available only for train and valid; test is forbidden")
        if limit is not None and limit <= 0:
            raise ValueError("history limit must be positive or None for all")
        rows = np.asarray(row_indices, dtype=np.int64)
        max_rows = len(self.train_user_slots) if split == "train" else len(self.valid_user_slots)
        if np.any(rows < 0) or np.any(rows >= max_rows):
            raise IndexError("history row index out of range")
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        chunks: list[np.ndarray] = []
        for output_row, row in enumerate(rows):
            slot = int(
                self.train_user_slots[row] if split == "train" else self.valid_user_slots[row]
            )
            if slot < 0:
                history = np.empty(0, dtype=np.int32)
            else:
                items = self.positive_items[slot]
                end = int(self.train_cutoffs[row]) if split == "train" else len(items)
                start = 0 if limit is None else max(0, end - limit)
                history = items[start:end]
            chunks.append(history)
            indptr[output_row + 1] = indptr[output_row] + len(history)
        flat = np.concatenate(chunks) if chunks and indptr[-1] else np.empty(0, dtype=np.int32)
        return indptr, flat

    def mean_pool(
        self,
        split: str,
        row_indices: Sequence[int] | np.ndarray,
        limit: int | None,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        indptr, items = self.batch_csr(split, row_indices, limit)
        return csr_mean_pool(indptr, items, embeddings)


def csr_mean_pool(indptr: np.ndarray, item_ids: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Mean pool ragged histories; empty rows are exact zero vectors."""
    indptr = np.asarray(indptr, dtype=np.int64)
    item_ids = np.asarray(item_ids, dtype=np.int64)
    if len(indptr) == 0 or indptr[0] != 0 or np.any(np.diff(indptr) < 0):
        raise ValueError("invalid CSR indptr")
    if int(indptr[-1]) != len(item_ids):
        raise ValueError("CSR indptr does not cover item_ids")
    if len(item_ids) and (np.any(item_ids < 0) or np.any(item_ids >= len(embeddings))):
        raise IndexError("history item id is outside embedding table")
    rows = len(indptr) - 1
    result = np.zeros((rows, embeddings.shape[1]), dtype=embeddings.dtype)
    lengths = np.diff(indptr)
    if len(item_ids):
        owners = np.repeat(np.arange(rows), lengths)
        np.add.at(result, owners, embeddings[item_ids])
        nonempty = lengths > 0
        result[nonempty] /= lengths[nonempty, None]
    return result


def _read_sidecars(data_dir: Path) -> dict[str, list[RawSidecarRow]]:
    output = {"train": [], "valid": []}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with (Path(data_dir) / filename).open(newline="") as stream:
            for row in csv.DictReader(stream):
                date = int(row["date"])
                for split in ("train", "valid"):
                    low, high = SPLITS[split]
                    if low <= date <= high:
                        output[split].append(
                            RawSidecarRow(
                                date=date,
                                user_id=row["user_id"],
                                video_id=row["video_id"],
                                time_ms=int(row["time_ms"]),
                            )
                        )
                        break
    return output


def build_causal_history(data_dir: Path, splits: dict, enc: dict) -> CausalPositiveHistory:
    """Build a compact index aligned exactly to official split row order."""
    raw = _read_sidecars(Path(data_dir))
    for split in ("train", "valid"):
        official = splits[split]
        if len(raw[split]) != len(official) or len(enc[split][0]) != len(official):
            raise ValueError(f"history sidecar row count mismatch for {split}")
        for index, (sidecar, row) in enumerate(zip(raw[split], official)):
            if (sidecar.date, sidecar.user_id, sidecar.video_id) != (row[0], row[1], row[2]):
                raise ValueError(f"history sidecar alignment mismatch at {split} row {index}")

    train_rows = splits["train"]
    train_items = enc["train"][0][:, 1].astype(np.int32, copy=False)
    events_by_user: dict[str, list[tuple[int, int, int]]] = {}
    for row_index, (sidecar, official, item) in enumerate(zip(raw["train"], train_rows, train_items)):
        if official[6] == 1:
            events_by_user.setdefault(sidecar.user_id, []).append(
                (sidecar.time_ms, row_index, int(item))
            )
    users = tuple(events_by_user)
    user_to_slot = {user: slot for slot, user in enumerate(users)}
    positive_times = []
    positive_items = []
    for user in users:
        events = sorted(events_by_user[user], key=lambda event: (event[0], event[1]))
        positive_times.append(np.asarray([event[0] for event in events], dtype=np.int64))
        positive_items.append(np.asarray([event[2] for event in events], dtype=np.int32))

    train_slots = np.empty(len(raw["train"]), dtype=np.int32)
    train_cutoffs = np.empty(len(raw["train"]), dtype=np.int32)
    for index, sidecar in enumerate(raw["train"]):
        slot = user_to_slot.get(sidecar.user_id, -1)
        train_slots[index] = slot
        train_cutoffs[index] = (
            0
            if slot < 0
            else int(np.searchsorted(positive_times[slot], sidecar.time_ms, side="left"))
        )
    valid_slots = np.asarray(
        [user_to_slot.get(sidecar.user_id, -1) for sidecar in raw["valid"]], dtype=np.int32
    )
    return CausalPositiveHistory(
        tuple(positive_times), tuple(positive_items), train_slots, train_cutoffs, valid_slots
    )
