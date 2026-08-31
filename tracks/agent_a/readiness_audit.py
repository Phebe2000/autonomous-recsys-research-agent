"""Read-only integrity audit for the shipped kit and audited KuaiRand state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import lightgbm
import numpy
import optuna
import scipy


FINGERPRINT_HEX = "6ea8eca058e1ae34c8a3748c4d4a633a853a761ae9420e1c86bd96e2878dd838"
EXPECTED_LEDGER_SHA256 = "ffbd841659fc7cf1b80752d01a1cffc607ee354f91856a087c4d356b8215d5d8"
EXPECTED_OFFICIAL_SHA256 = {
    "data.py": "1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541",
    "baseline.py": "c8f7fc60178413e247e78bb231e7550eeef52101b6493fcf1a4d2b0e5fe18f8a",
    "evaluate.py": "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de",
    "submit.py": "ab01bb2b970ae2a9f2ead299f5240b71ff4126c2d9bb0e0c4de6c7e245dc148c",
}
EXPECTED_PRIMARY = 0.6015123724937439


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(repo_root: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    official = {name: _digest(repo_root / name) for name in EXPECTED_OFFICIAL_SHA256}
    if official != EXPECTED_OFFICIAL_SHA256:
        raise RuntimeError("official starter-kit SHA-256 integrity check failed")
    state_dir = repo_root / "tracks" / "agent_a" / "runtime" / FINGERPRINT_HEX
    ledger = state_dir / "research.sqlite3"
    if _digest(ledger) != EXPECTED_LEDGER_SHA256:
        raise RuntimeError("audited KuaiRand ledger bytes changed")
    connection = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        used = int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    finally:
        connection.close()
    top1 = json.loads((state_dir / "top1.json").read_text())
    if used != 24:
        raise RuntimeError(f"audited KuaiRand budget changed: {used}/50")
    if top1["trial"]["trial_id"] != "trial-15":
        raise RuntimeError("audited KuaiRand Top-1 trial changed")
    if abs(float(top1["validation"]["primary"]) - EXPECTED_PRIMARY) > 1e-15:
        raise RuntimeError("audited KuaiRand Top-1 primary changed")
    if top1.get("test_metrics_used") is not False:
        raise RuntimeError("audited Top-1 is not validation-only")
    return {
        "agent_dependencies": {
            "lightgbm": lightgbm.__version__,
            "numpy": numpy.__version__,
            "optuna": optuna.__version__,
            "scipy": scipy.__version__,
        },
        "official_sha256": official,
        "kuairand": {
            "dataset_fingerprint": f"sha256:{FINGERPRINT_HEX}",
            "ledger_sha256": EXPECTED_LEDGER_SHA256,
            "used": used,
            "remaining": 50 - used,
            "top1_trial_id": "trial-15",
            "top1_validation_primary": EXPECTED_PRIMARY,
            "test_metrics_used": False,
        },
        "read_only": True,
        "status": "ready",
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    print(json.dumps(audit(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
