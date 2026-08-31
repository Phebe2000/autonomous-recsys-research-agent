"""SQLite research ledger with an atomic, non-resettable 50-trial budget."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .contracts import TrialOutcome, reject_test_data
from .fingerprint import canonical_json

MAX_TRIALS = 50


class BudgetExhausted(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchStore:
    """One persistent ledger per dataset fingerprint.

    Reservation is committed before candidate execution. Every row, including
    reserved/running/failed/pruned, consumes budget permanently.
    """

    def __init__(self, path: Path, dataset_fingerprint: str, max_trials: int = MAX_TRIALS):
        if not 1 <= max_trials <= MAX_TRIALS:
            raise ValueError(f"max_trials must be in [1, {MAX_TRIALS}]")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_fingerprint = dataset_fingerprint
        self.max_trials = max_trials
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS trials(
                    ordinal INTEGER PRIMARY KEY,
                    trial_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    method TEXT NOT NULL,
                    hypothesis TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    validation_json TEXT,
                    validation_primary REAL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(ordinal BETWEEN 1 AND 50)
                );
                CREATE TABLE IF NOT EXISTS memory_events(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            existing = dict(db.execute("SELECT key,value FROM metadata").fetchall())
            expected = {
                "schema_version": "1",
                "dataset_fingerprint": self.dataset_fingerprint,
                "max_trials": str(self.max_trials),
            }
            if existing:
                for key, value in expected.items():
                    if existing.get(key) != value:
                        raise ValueError(f"persistent ledger metadata mismatch for {key}")
            else:
                db.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", expected.items())

    @property
    def consumed(self) -> int:
        with self._read() as db:
            return int(db.execute("SELECT COUNT(*) FROM trials").fetchone()[0])

    @property
    def remaining(self) -> int:
        return self.max_trials - self.consumed

    def reserve(self, method: str, hypothesis: str, config: dict[str, Any], seed: int = 0) -> dict:
        reject_test_data(config, "config")
        config_json = canonical_json(config)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        with self._transaction() as db:
            count = int(db.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
            if count >= self.max_trials:
                raise BudgetExhausted(f"dataset trial budget exhausted ({self.max_trials}/{self.max_trials})")
            ordinal = count + 1
            trial_id = f"trial-{ordinal:02d}"
            now = _now()
            db.execute(
                """INSERT INTO trials
                (ordinal,trial_id,status,method,hypothesis,seed,config_json,config_hash,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (ordinal, trial_id, "reserved", method, hypothesis, seed, config_json, config_hash, now, now),
            )
            self._event(db, trial_id, "trial_reserved", {"ordinal": ordinal, "config_hash": config_hash})
        return self.get(trial_id)

    def mark_running(self, trial_id: str) -> dict:
        with self._transaction() as db:
            self._require_status(db, trial_id, {"reserved", "running"})
            db.execute("UPDATE trials SET status='running',updated_at=? WHERE trial_id=?", (_now(), trial_id))
            self._event(db, trial_id, "trial_started", {})
        return self.get(trial_id)

    def complete(self, trial_id: str, outcome: TrialOutcome) -> dict:
        result = outcome.to_dict()
        reject_test_data(result)
        validation = outcome.validation.to_dict()
        with self._transaction() as db:
            self._require_status(db, trial_id, {"reserved", "running"})
            db.execute(
                """UPDATE trials SET status='completed',validation_json=?,validation_primary=?,
                result_json=?,updated_at=? WHERE trial_id=?""",
                (canonical_json(validation), outcome.validation.primary, canonical_json(result), _now(), trial_id),
            )
            self._event(db, trial_id, "trial_completed", {"validation": validation})
        return self.get(trial_id)

    def fail(self, trial_id: str, error: str, pruned: bool = False) -> dict:
        status = "pruned" if pruned else "failed"
        with self._transaction() as db:
            self._require_status(db, trial_id, {"reserved", "running"})
            db.execute(
                "UPDATE trials SET status=?,error=?,updated_at=? WHERE trial_id=?",
                (status, str(error), _now(), trial_id),
            )
            self._event(db, trial_id, f"trial_{status}", {"error": str(error)})
        return self.get(trial_id)

    def resume(self, trial_id: str) -> dict:
        """Return an existing reservation without consuming another trial."""
        trial = self.get(trial_id)
        if trial["status"] not in {"reserved", "running"}:
            raise ValueError("only reserved or running trials can resume")
        return trial

    def add_note(self, message: str, trial_id: str | None = None) -> None:
        with self._transaction() as db:
            self._event(db, trial_id, "note", {"message": message})

    def get(self, trial_id: str) -> dict:
        with self._read() as db:
            row = db.execute("SELECT * FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        return self._decode(row)

    def trials(self) -> list[dict]:
        with self._read() as db:
            rows = db.execute("SELECT * FROM trials ORDER BY ordinal").fetchall()
        return [self._decode(row) for row in rows]

    def best_trial(self) -> dict | None:
        with self._read() as db:
            row = db.execute(
                """SELECT * FROM trials WHERE status='completed' AND validation_primary IS NOT NULL
                ORDER BY validation_primary DESC, ordinal ASC LIMIT 1"""
            ).fetchone()
        return None if row is None else self._decode(row)

    def _require_status(self, db: sqlite3.Connection, trial_id: str, allowed: set[str]) -> None:
        row = db.execute("SELECT status FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        if row["status"] not in allowed:
            raise ValueError(f"trial {trial_id} has status {row['status']}, expected {sorted(allowed)}")

    def _event(self, db, trial_id: str | None, kind: str, payload: dict) -> None:
        db.execute(
            "INSERT INTO memory_events(trial_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
            (trial_id, kind, canonical_json(payload), _now()),
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        result = dict(row)
        for key in ("config_json", "validation_json", "result_json"):
            result[key.removesuffix("_json")] = None if result[key] is None else json.loads(result[key])
            del result[key]
        return result
