"""Content-addressed onboarding for per-dataset budget isolation."""

from __future__ import annotations

import hashlib
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
