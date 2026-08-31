"""Validate hidden-test submission alignment without loading relevance labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from submit import HEADER

from .safe_data import load_unlabeled_exposures


def check_submission(path: Path, data_dir: Path, split: str = "test") -> dict:
    exposures = load_unlabeled_exposures(data_dir, split)
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != HEADER:
            raise ValueError(f"submission header must be {HEADER}")
        count = 0
        for line, record in enumerate(reader, start=2):
            if len(record) != 4:
                raise ValueError(f"submission line {line} must contain four fields")
            if count >= len(exposures):
                raise ValueError("submission has more rows than the evaluation split")
            row_id, user_id, video_id, score = record
            expected = exposures[count]
            if int(row_id) != count:
                raise ValueError(f"submission row_id gap at line {line}")
            if (user_id, video_id) != (expected.user_id, expected.video_id):
                raise ValueError(f"submission exposure misalignment at line {line}")
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"submission score is non-finite at line {line}")
            count += 1
    if count != len(exposures):
        raise ValueError(f"submission has {count} rows; expected {len(exposures)}")
    return {
        "schema_valid": True,
        "split": split,
        "rows": count,
        "hidden_labels_loaded": False,
        "test_metrics_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    print(json.dumps(
        check_submission(Path(args.path), Path(args.data_dir), args.split),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
