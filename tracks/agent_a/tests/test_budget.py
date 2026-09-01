from pathlib import Path
import tempfile
import unittest

from tracks.agent_a.store import BudgetExhausted, ResearchStore


class BudgetTest(unittest.TestCase):
    def test_hard_cap_counts_failed_pruned_and_running(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite3"
            store = ResearchStore(path, "sha256:dataset-a")
            for index in range(47):
                store.reserve("method", f"trial {index}", {"index": index})
            failed = store.reserve("method", "failed", {})
            store.fail(failed["trial_id"], "boom")
            pruned = store.reserve("method", "pruned", {})
            store.fail(pruned["trial_id"], "unpromising", pruned=True)
            running = store.reserve("method", "running", {})
            store.mark_running(running["trial_id"])
            self.assertEqual(store.consumed, 50)
            self.assertEqual(store.remaining, 0)
            with self.assertRaises(BudgetExhausted):
                store.reserve("method", "51st", {})

    def test_resume_does_not_reset_or_consume_again(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite3"
            first = ResearchStore(path, "sha256:dataset-a")
            trial = first.reserve("method", "resume me", {})
            first.mark_running(trial["trial_id"])
            reopened = ResearchStore(path, "sha256:dataset-a")
            self.assertEqual(reopened.resume(trial["trial_id"])["trial_id"], trial["trial_id"])
            self.assertEqual(reopened.consumed, 1)
            with self.assertRaises(ValueError):
                ResearchStore(path, "sha256:different-dataset")

    def test_fingerprint_namespaces_have_independent_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = ResearchStore(root / "a.sqlite3", "sha256:a")
            b = ResearchStore(root / "b.sqlite3", "sha256:b")
            a.reserve("method", "a", {})
            self.assertEqual(a.remaining, 49)
            self.assertEqual(b.remaining, 50)


if __name__ == "__main__":
    unittest.main()
