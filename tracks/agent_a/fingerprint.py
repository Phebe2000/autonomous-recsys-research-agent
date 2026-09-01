"""Content-addressed onboarding for per-dataset budget isolation."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any

SOURCE_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)

SEMANTICS = {
    "loader": "official-data.py-v1",
    "label": "long_view",
    "splits": {
        "train": [20220408, 20220421],
        "valid": [20220422, 20220428],
        "test": [20220429, 20220508],
    },
    "row_order": "official-file-order-then-date-filter-v1",
    "task": "within-user-logged-exposure-ranking",
}

JUDGED_INFERENCE_COLUMNS = (
    "user_id", "video_id", "date", "hourmin", "time_ms",
    "duration_ms", "is_rand", "tab",
)
JUDGED_TRAIN_COLUMNS = (
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click",
    "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
)
JUDGED_VALID_COLUMNS = JUDGED_INFERENCE_COLUMNS + ("long_view",)


def canonical_json(value: Any) -> str:
    def convert(item):
        if hasattr(item, "item"):
            return item.item()
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(f"{type(item).__name__} is not JSON serializable")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=convert,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dataset(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir).resolve()
    files = []
    for name in SOURCE_FILES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file is missing: {path}")
        files.append(
            {"relative_name": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    identity = {"source_files": files, "semantics": SEMANTICS}
    fingerprint = "sha256:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return {
        "schema_version": 1,
        "fingerprint_algorithm": "sha256-canonical-v1",
        "dataset_fingerprint": fingerprint,
        **identity,
    }


def _judged_log_digest(path: Path) -> tuple[str, int]:
    """Hash only columns allowed to affect a judged run.

    Train may use every feedback field, validation contributes its public target,
    and hidden-test rows contribute inference fields only. Test-label mutations
    therefore cannot alter research identity or search state.
    """
    digest = hashlib.sha256()
    count = 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            if SEMANTICS["splits"]["train"][0] <= date <= SEMANTICS["splits"]["train"][1]:
                split, columns = "train", JUDGED_TRAIN_COLUMNS
            elif SEMANTICS["splits"]["valid"][0] <= date <= SEMANTICS["splits"]["valid"][1]:
                split, columns = "valid", JUDGED_VALID_COLUMNS
            elif SEMANTICS["splits"]["test"][0] <= date <= SEMANTICS["splits"]["test"][1]:
                split, columns = "test", JUDGED_INFERENCE_COLUMNS
            else:
                continue
            digest.update(canonical_json({"split": split, "values": [row[name] for name in columns]}).encode())
            digest.update(b"\n")
            count += 1
    return digest.hexdigest(), count


def fingerprint_judged_dataset(data_dir: Path) -> dict[str, Any]:
    """Content identity for a KuaiRand-Pure judged run without test labels.

    The randomized-exposure log is deliberately excluded: it is EDA-only and
    cannot influence training identity. Static feature files are permissible
    benchmark inputs and are hashed byte-for-byte.
    """
    data_dir = Path(data_dir).resolve()
    sources = []
    for name in SOURCE_FILES[:2]:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file is missing: {path}")
        digest, rows = _judged_log_digest(path)
        sources.append({"relative_name": name, "permitted_projection_sha256": digest, "rows": rows})
    for name in SOURCE_FILES[3:]:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"required dataset file is missing: {path}")
        sources.append({"relative_name": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    identity = {
        "benchmark": "KuaiRand-Pure",
        "source_files": sources,
        "excluded_from_training_identity": ["log_random_4_22_to_5_08_pure.csv"],
        "semantics": {**SEMANTICS, "loader": "agent-a-judged-label-boundary-v1"},
    }
    fingerprint = "sha256:" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return {
        "schema_version": 1,
        "fingerprint_algorithm": "sha256-judged-permitted-columns-v1",
        "dataset_fingerprint": fingerprint,
        **identity,
    }
