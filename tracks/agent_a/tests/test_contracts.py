from unittest import mock
import unittest

import numpy as np

from tracks.agent_a.contracts import ContractError, ValidationMetrics, reject_test_data
from tracks.agent_a.fingerprint import canonical_json
from tracks.agent_a.runner import TrialRunner, TrialSpec
from tracks.agent_a.store import ResearchStore
from pathlib import Path
import tempfile
import json
from tracks.agent_a.guards import evaluate_checked


class ContractTest(unittest.TestCase):
    def test_primary_formula(self):
        metrics = ValidationMetrics.from_mapping({"GAUC": 0.2, "nDCG@5": 0.8, "primary": 0.5})
        self.assertEqual(metrics.primary, 0.5)
        with self.assertRaises(ContractError):
            ValidationMetrics.from_mapping({"GAUC": 0.2, "nDCG@5": 0.8, "primary": 0.6})
        with self.assertRaises(ContractError):
            ValidationMetrics.from_mapping({"GAUC": float("nan"), "nDCG@5": 0.8})

    def test_rejects_test_metrics_at_any_depth(self):
        for payload in (
            {"test": {"primary": 1.0}},
            {"test_primary": 1.0},
            {"history": [{"metrics_test": 1.0}]},
        ):
            with self.assertRaises(ContractError):
                reject_test_data(payload)

    def test_canonical_json_normalizes_numpy_diagnostics(self):
        self.assertEqual(canonical_json({"loss": np.float32(0.25)}), '{"loss":0.25}')

    def test_length_guard_runs_before_official_evaluator(self):
        cases = ((["u", "u"], [0], [0.1, 0.2]), (["u"], [0, 1], [0.1]), (["u"], [0], [0.1, 0.2]))
        with mock.patch("tracks.agent_a.guards.official_evaluate") as evaluator:
            for users, labels, scores in cases:
                with self.assertRaises(ContractError):
                    evaluate_checked(users, labels, scores)
            evaluator.assert_not_called()

    def test_aligned_guard_delegates_without_reordering(self):
        expected = {"GAUC": 0.5, "nDCG@5": 0.5, "primary": 0.5, "users": 1, "rows": 2}
        with mock.patch("tracks.agent_a.guards.official_evaluate", return_value=expected) as evaluator:
            result = evaluate_checked(("u", "u"), (0, 1), (0.1, 0.2))
        self.assertEqual(result, expected)
        evaluator.assert_called_once_with(("u", "u"), (0, 1), (0.1, 0.2), k=5)

    def test_invalid_candidate_result_is_failed_and_consumes_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "ledger.sqlite3", "sha256:dataset")
            runner = TrialRunner(store)

            def invalid(_trial):
                raise ContractError("test metrics are forbidden")

            with self.assertRaises(ContractError):
                runner.execute(TrialSpec("bad", "reject", {}), invalid)
            self.assertEqual(store.consumed, 1)
            self.assertEqual(store.trials()[0]["status"], "failed")

    def test_prior_contains_no_transferred_checkpoint_embedding_or_score(self):
        prior_path = Path(__file__).parents[1] / "configs" / "listnet_prior.json"
        prior = json.loads(prior_path.read_text())
        serialized = json.dumps(prior).lower()
        for forbidden in ("checkpoint_path", "embedding_path", "valid_primary", "test_primary"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(prior["initialization"], "fresh_validation_best_official_fm_on_current_dataset")


if __name__ == "__main__":
    unittest.main()
