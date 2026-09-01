import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import tempfile
import unittest

import numpy as np

from tracks.agent_a.autonomous import AutonomousResearchLoop, SyntheticExecutor
from tracks.agent_a.compliance import JudgedRunCompliance, JudgedRunPolicy, WallClockExceeded
from tracks.agent_a.contracts import TrialOutcome, ValidationMetrics
from tracks.agent_a.finalize import score_locked_exposures
from tracks.agent_a.fingerprint import fingerprint_judged_dataset
from tracks.agent_a.judged_cli import _active, initialize, plan, supersede_empty_preflight
from tracks.agent_a.safe_data import (
    encode_research_splits,
    load_research_splits,
    load_unlabeled_exposures,
)
from tracks.agent_a.safe_submit_check import check_submission
from tracks.agent_a.store import ResearchStore
from tracks.agent_a.tests.test_readiness import write_fixture_dataset


def flip_test_labels(data_dir: Path) -> None:
    path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if int(row["date"]) >= 20220429:
            row["long_view"] = "0" if row["long_view"] != "0" else "1"
            row["is_click"] = "0" if row["is_click"] != "0" else "1"
            row["play_time_ms"] = str(int(row["play_time_ms"]) + 999999)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SafeDataBoundaryTest(unittest.TestCase):
    def test_hidden_label_mutation_cannot_change_fingerprint_or_research_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            write_fixture_dataset(data_dir)
            before_fingerprint = fingerprint_judged_dataset(data_dir)
            before_splits = load_research_splits(data_dir)
            before_enc, _ = encode_research_splits(before_splits)
            before_test = load_unlabeled_exposures(data_dir, "test")
            self.assertEqual(set(before_splits), {"train", "valid"})

            flip_test_labels(data_dir)
            after_fingerprint = fingerprint_judged_dataset(data_dir)
            after_splits = load_research_splits(data_dir)
            after_enc, _ = encode_research_splits(after_splits)
            after_test = load_unlabeled_exposures(data_dir, "test")

            self.assertEqual(before_fingerprint, after_fingerprint)
            self.assertEqual(before_splits, after_splits)
            self.assertEqual(before_test, after_test)
            for split in ("train", "valid"):
                for before, after in zip(before_enc[split][:2], after_enc[split][:2]):
                    np.testing.assert_array_equal(before, after)

    def test_random_log_is_not_part_of_judged_training_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            write_fixture_dataset(data_dir)
            before = fingerprint_judged_dataset(data_dir)["dataset_fingerprint"]
            random_log = data_dir / "log_random_4_22_to_5_08_pure.csv"
            random_log.write_text(random_log.read_text() + "arbitrary,eda,only\n")
            after = fingerprint_judged_dataset(data_dir)["dataset_fingerprint"]
            self.assertEqual(before, after)


class CompliancePolicyTest(unittest.TestCase):
    def test_recorded_stop_freezes_wall_clock_for_later_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = [datetime.now(timezone.utc)]
            fingerprint = "sha256:synthetic-frozen-wall-clock"
            policy = JudgedRunPolicy(fingerprint, minimum_scored_iterations=4)
            compliance = JudgedRunCompliance(root, policy, now=lambda: current[0])
            store = ResearchStore(root / "research.sqlite3", fingerprint)
            compliance.start_if_needed()
            store.record_agent_decision(
                stage="stopping",
                decision="stop_research_loop",
                rationale="fixture convergence",
                evidence={"stop_reason": "fixture_converged"},
                alternatives=("continue_beyond_policy",),
                selected_action="stop",
                actor="fixture",
                decision_key="stop:fixture",
            )
            first = compliance.write_audit(store)
            current[0] += timedelta(hours=7)
            later = compliance.write_audit(store)
            self.assertEqual(
                later["timing"]["elapsed_seconds"],
                first["timing"]["elapsed_seconds"],
            )
            self.assertFalse(later["timing"]["wall_clock_exhausted"])
            self.assertEqual(later["timing"]["stop_reason"], "fixture_converged")

    def test_agent_decisions_are_idempotent_hash_chained_and_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = "sha256:synthetic-decision-journal"
            compliance = JudgedRunCompliance(root, JudgedRunPolicy(fingerprint))
            store = ResearchStore(root / "research.sqlite3", fingerprint)
            trial = store.reserve("fixture", "decision fixture", {
                "research_action": {
                    "code_diff": "",
                    "code_diff_sha256": __import__("hashlib").sha256(b"").hexdigest(),
                }
            })
            first = store.record_agent_decision(
                stage="candidate_selection",
                decision="execute_candidate",
                rationale="Validation-safe fixture candidate is ready.",
                evidence={"validation_available": True},
                selected_action="train_and_validate",
                alternatives=("skip_candidate",),
                trial_id=trial["trial_id"],
                decision_key="fixture:execute",
            )
            duplicate = store.record_agent_decision(
                stage="candidate_selection",
                decision="execute_candidate",
                rationale="Validation-safe fixture candidate is ready.",
                evidence={"validation_available": True},
                selected_action="train_and_validate",
                alternatives=("skip_candidate",),
                trial_id=trial["trial_id"],
                decision_key="fixture:execute",
            )
            self.assertEqual(first, duplicate)
            second = store.record_agent_decision(
                stage="candidate_disposition",
                decision="retain_evidence",
                rationale="The candidate has not become validation Top-1.",
                evidence={"validation_primary": 0.5},
                selected_action="keep_current_top1",
                alternatives=("promote_without_evidence",),
                trial_id=trial["trial_id"],
                decision_key="fixture:disposition",
            )
            self.assertEqual(second["previous_decision_sha256"], first["decision_sha256"])
            self.assertEqual(len(store.decisions()), 2)
            with self.assertRaisesRegex(ValueError, "test data"):
                store.record_agent_decision(
                    stage="invalid",
                    decision="reject",
                    rationale="Forbidden evidence must fail.",
                    evidence={"test_metric": 0.9},
                    selected_action="fail",
                )
            audit = compliance.write_audit(store)
            journal = json.loads((root / "agent_decision_journal.json").read_text())
            self.assertEqual(journal["decision_count"], 2)
            self.assertEqual(
                journal["final_decision_sha256"], second["decision_sha256"]
            )
            self.assertEqual(audit["agent_decision_journal"]["decision_count"], 2)

    def test_policy_is_immutable_and_wall_clock_persists_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = "sha256:synthetic-compliance"
            current = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
            policy = JudgedRunPolicy(fingerprint, wall_clock_seconds=60)
            compliance = JudgedRunCompliance(root, policy, now=lambda: current[0])
            loop = AutonomousResearchLoop(
                root, fingerprint, simulation=True, max_trials=50,
                convergence_enabled=False, compliance=compliance,
            )
            loop.step(SyntheticExecutor())
            self.assertEqual(loop.store.consumed, 1)
            with self.assertRaises(WallClockExceeded):
                compliance.execute_with_deadline(lambda: signal.raise_signal(signal.SIGALRM))
            current[0] += timedelta(seconds=60)
            resumed_compliance = JudgedRunCompliance(root, policy, now=lambda: current[0])
            resumed = AutonomousResearchLoop(
                root, fingerprint, simulation=True, max_trials=50,
                convergence_enabled=False, compliance=resumed_compliance,
            )
            self.assertEqual(resumed.stop_reason(), "wall_clock_exhausted")
            self.assertEqual(resumed.step(SyntheticExecutor())["stop_reason"], "wall_clock_exhausted")
            self.assertEqual(resumed.store.consumed, 1)
            with self.assertRaisesRegex(ValueError, "immutable"):
                JudgedRunCompliance(
                    root,
                    JudgedRunPolicy(fingerprint, epsilon=0.001, wall_clock_seconds=60),
                    now=lambda: current[0],
                )

    def test_cumulative_window_uses_lte_and_failures_do_not_advance_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = "sha256:synthetic-convergence-compliance"
            policy = JudgedRunPolicy(
                fingerprint, epsilon=0.002, convergence_n=3,
                minimum_scored_iterations=4,
            )
            compliance = JudgedRunCompliance(root, policy)
            loop = AutonomousResearchLoop(
                root, fingerprint, simulation=True, max_trials=50,
                compliance=compliance,
            )
            values = [0.600, 0.601, 0.602, 0.602]
            for index, value in enumerate(values):
                if index == 2:
                    failed = loop.store.reserve("fixture", "crash", {"index": "failure"})
                    loop.store.fail(failed["trial_id"], "intentional")
                    self.assertIsNone(loop.stop_reason())
                trial = loop.store.reserve("fixture", "scored", {"index": index})
                loop.store.complete(
                    trial["trial_id"],
                    TrialOutcome(ValidationMetrics(value, value, value), 1, "fixture"),
                )
            self.assertIn("converged_no_0.002_gain_for_3", loop.stop_reason())

    def test_audit_requires_and_records_code_diff_and_recovery_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = "sha256:synthetic-audit-compliance"
            policy = JudgedRunPolicy(fingerprint)
            compliance = JudgedRunCompliance(root, policy)
            store = ResearchStore(root / "standalone.sqlite3", fingerprint)
            missing = store.reserve("fixture", "missing diff", {})
            store.fail(missing["trial_id"], "boom")
            with self.assertRaisesRegex(ValueError, "code diff"):
                compliance.write_audit(store)

            other_root = root / "valid"
            valid_compliance = JudgedRunCompliance(other_root, policy)
            valid_store = ResearchStore(other_root / "research.sqlite3", fingerprint)
            diff = "+ safe change\n"
            trial = valid_store.reserve(
                "fixture", "recorded hypothesis",
                {"research_action": {
                    "code_diff": diff,
                    "code_diff_sha256": __import__("hashlib").sha256(diff.encode()).hexdigest(),
                    "change_kind": "generated_patch",
                }},
            )
            valid_store.fail(trial["trial_id"], "recoverable error")
            audit = valid_compliance.write_audit(valid_store)
            self.assertEqual(audit["iterations"][0]["code_diff"], diff)
            self.assertEqual(audit["iterations"][0]["error"], "recoverable error")
            self.assertTrue(audit["iterations"][0]["events"])


class JudgedRunSmokeTest(unittest.TestCase):
    def test_unstarted_preflight_can_be_auditably_superseded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, state_root = root / "data", root / "judged"
            write_fixture_dataset(data_dir)
            initialize(data_dir, state_root)
            archived = supersede_empty_preflight(
                data_dir, state_root, "agent implementation changed before iteration one"
            )
            self.assertEqual(archived["budget_used"], 0)
            self.assertFalse(archived["clock_started"])
            self.assertTrue(Path(archived["archive"]).joinpath("superseded.json").exists())
            self.assertFalse(plan(data_dir, state_root)["initialized"])
            fresh = initialize(data_dir, state_root)
            self.assertEqual(fresh["budget"]["used"], 0)

    def test_clean_zero_budget_resume_lock_and_hidden_label_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            state_root = root / "judged"
            write_fixture_dataset(data_dir)
            self.assertEqual(plan(data_dir, state_root)["budget"]["used"], 0)
            initialized = initialize(data_dir, state_root)
            self.assertEqual(initialized["budget"], {"used": 0, "remaining": 50, "maximum": 50})
            self.assertFalse(initialized["clock_started"])

            _, compliance, loop = _active(data_dir, state_root)
            from tracks.agent_a.real_executor import RealCandidateExecutor
            executor = RealCandidateExecutor(data_dir, loop.state_dir, loop.store, loop.fingerprint)
            loop.run(executor, target_trials=3)
            _, resumed_compliance, resumed = _active(data_dir, state_root)
            resumed_executor = RealCandidateExecutor(
                data_dir, resumed.state_dir, resumed.store, resumed.fingerprint
            )
            resumed.run(resumed_executor, target_trials=6)
            self.assertEqual(resumed.store.consumed, 6)
            audit = resumed_compliance.write_audit(resumed.store)
            self.assertEqual(len(audit["iterations"]), 6)
            self.assertTrue(all("code_diff" in item for item in audit["iterations"]))
            resumed.lock()
            decisions = resumed.store.decisions()
            self.assertGreaterEqual(len(decisions), 13)
            self.assertEqual(decisions[-1]["payload"]["decision"], "lock_validation_best_checkpoint")

            first = root / "first.csv"
            score_locked_exposures(data_dir, state_root, first, judged=True)
            first_bytes = first.read_bytes()
            fingerprint = resumed.fingerprint
            flip_test_labels(data_dir)
            self.assertEqual(
                fingerprint_judged_dataset(data_dir)["dataset_fingerprint"], fingerprint
            )
            second = root / "second.csv"
            result = score_locked_exposures(data_dir, state_root, second, judged=True)
            self.assertEqual(second.read_bytes(), first_bytes)
            self.assertFalse(result["hidden_labels_loaded"])
            refreshed = json.loads((resumed.state_dir / "agent_decision_journal.json").read_text())
            self.assertTrue(any(
                event["payload"]["decision"] == "emit_locked_exposure_scores"
                for event in refreshed["decisions"]
            ))
            checked = check_submission(second, data_dir, "test")
            self.assertTrue(checked["schema_valid"])
            self.assertFalse(checked["hidden_labels_loaded"])
            with second.open(newline="") as stream:
                self.assertEqual(len(list(csv.reader(stream))) - 1, len(load_unlabeled_exposures(data_dir, "test")))


if __name__ == "__main__":
    unittest.main()
