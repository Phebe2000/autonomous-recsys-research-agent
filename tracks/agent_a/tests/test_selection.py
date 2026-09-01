import json
from pathlib import Path
import tempfile
import unittest

from tracks.agent_a.contracts import TrialOutcome, ValidationMetrics
from tracks.agent_a.selection import write_top1
from tracks.agent_a.store import ResearchStore


def outcome(primary: float) -> TrialOutcome:
    return TrialOutcome(
        validation=ValidationMetrics(primary, primary, primary),
        best_step=1,
        stop_reason="done",
    )


class SelectionTest(unittest.TestCase):
    def test_top1_uses_validation_primary_only_with_stable_tie_break(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResearchStore(root / "ledger.sqlite3", "sha256:dataset")
            first = store.reserve("baseline", "first", {"name": "first"})
            store.complete(first["trial_id"], outcome(0.7))
            best = store.reserve("listnet", "best", {"name": "best"})
            store.complete(best["trial_id"], outcome(0.8))
            tie = store.reserve("later", "tie", {"name": "tie"})
            store.complete(tie["trial_id"], outcome(0.8))
            failed = store.reserve("failed", "failed", {})
            store.fail(failed["trial_id"], "ignored")
            manifest = write_top1(store, root / "top1.json")
            self.assertEqual(manifest["trial"]["trial_id"], best["trial_id"])
            self.assertEqual(manifest["validation"]["primary"], 0.8)
            self.assertEqual(manifest["selection"]["split"], "valid")
            self.assertFalse(manifest["test_metrics_used"])
            reloaded = json.loads((root / "top1.json").read_text())
            self.assertEqual(reloaded["trial"]["trial_id"], best["trial_id"])

    def test_vertical_baseline_to_listnet_manifest_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ledger.sqlite3"
            store = ResearchStore(path, "sha256:dataset")
            baseline = store.reserve("baseline", "baseline", {})
            store.complete(baseline["trial_id"], outcome(0.6))
            listnet = store.reserve("listnet", "prior", {})
            store.complete(listnet["trial_id"], outcome(0.61))
            reopened = ResearchStore(path, "sha256:dataset")
            manifest = write_top1(reopened, root / "top1.json")
            self.assertEqual(reopened.consumed, 2)
            self.assertEqual(manifest["trial"]["method"], "listnet")


if __name__ == "__main__":
    unittest.main()
