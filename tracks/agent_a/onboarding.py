"""Create immutable dataset metadata and its fingerprint-isolated state directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from data import load

from .fingerprint import fingerprint_dataset
from .store import ResearchStore


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def onboard(data_dir: Path, state_root: Path, purpose: str = "development") -> dict:
    identity = fingerprint_dataset(data_dir)
    splits = load(str(data_dir))
    fingerprint = identity["dataset_fingerprint"]
    state_dir = Path(state_root) / fingerprint.removeprefix("sha256:")
    manifest = {
        **identity,
        "purpose": purpose,
        "source_root": str(Path(data_dir).resolve()),
        "observed_rows": {name: len(rows) for name, rows in splits.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = state_dir / "dataset.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["dataset_fingerprint"] != fingerprint:
            raise ValueError("existing state directory belongs to another dataset")
        manifest = existing
    else:
        _atomic_json(manifest_path, manifest)
    store = ResearchStore(state_dir / "research.sqlite3", fingerprint)
    store.add_note(f"dataset onboarded for purpose={purpose}")
    return {"manifest": manifest, "state_dir": str(state_dir), "ledger": str(store.path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="KuaiRand-Pure/data")
    parser.add_argument("--state-root", default="tracks/agent_a/runtime")
    parser.add_argument("--purpose", default="development")
    args = parser.parse_args()
    print(json.dumps(onboard(Path(args.data_dir), Path(args.state_root), args.purpose), indent=2))


if __name__ == "__main__":
    main()
