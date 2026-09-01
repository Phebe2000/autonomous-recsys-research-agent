import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from data import encode, load
from submit import read_submission

from tracks.agent_a.autonomous import AutonomousResearchLoop
from tracks.agent_a.autonomous_cli import inspect, plan
from tracks.agent_a.fingerprint import fingerprint_dataset
from tracks.agent_a.finalize import (
    frozen_train_positive_history_csr,
    score_locked_exposures,
)
from tracks.agent_a.real_executor import RealCandidateExecutor
from tracks.agent_a.store import BudgetExhausted


LOG_HEADER = [
    "user_id", "video_id", "date", "hourmin", "time_ms", "is_click",
    "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "duration_ms", "profile_stay_time",
    "comment_stay_time", "is_profile_enter", "is_rand", "tab",
]


def write_fixture_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    train = []
    later = []
    for offset, (video, label) in enumerate((("v1", 1), ("v2", 0), ("v3", 1), ("v4", 0))):
        train.append([
            "u1", video, 20220410, 1200, 1000 + offset, label, 0, 0, 0, 0, 0,
            label, 1000 + 100 * offset, 2000, 0, 0, 0, 0, "tab",
        ])
        later.append([
            "u1", video, 20220422, 1200, 2000 + offset, label, 0, 0, 0, 0, 0,
            label, 1000 + 100 * offset, 2000, 0, 0, 0, 0, "tab",
        ])
        later.append([
            "u1", video, 20220429, 1200, 3000 + offset, label, 0, 0, 0, 0, 0,
            label, 1000 + 100 * offset, 2000, 0, 0, 0, 0, "tab",
        ])
    for filename, rows in (
        ("log_standard_4_08_to_4_21_pure.csv", train),
        ("log_standard_4_22_to_5_08_pure.csv", later),
        ("log_random_4_22_to_5_08_pure.csv", []),
    ):
        with (root / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(LOG_HEADER)
            writer.writerows(rows)
    with (root / "video_features_basic_pure.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "author_id"])
        writer.writerows([[f"v{index}", f"a{index}"] for index in range(1, 5)])
    for filename in ("user_features_pure.csv", "video_features_statistic_pure.csv"):
        (root / filename).write_text("fixture\n")


class RealExecutorReadinessTest(unittest.TestCase):
    def test_temporary_real_format_plan_run_resume_inspect_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            state_root = root / "production-fixture-state"
            write_fixture_dataset(data_dir)
            fingerprint = fingerprint_dataset(data_dir)["dataset_fingerprint"]
            self.assertFalse(fingerprint.startswith("sha256:synthetic-"))
            before = plan(data_dir, state_root)
            self.assertEqual(before["budget"]["used"], 0)
            self.assertFalse((state_root / fingerprint.removeprefix("sha256:") / "research.sqlite3").exists())

            state_dir = state_root / fingerprint.removeprefix("sha256:")
            first = AutonomousResearchLoop(
                state_dir, fingerprint, simulation=False,
                max_trials=50, convergence_enabled=False,
            )
            first_executor = RealCandidateExecutor(data_dir, state_dir, first.store, fingerprint)
            first.run(first_executor, target_trials=3)
            self.assertEqual(first.store.consumed, 3)

            resumed = AutonomousResearchLoop(
                state_dir, fingerprint, simulation=False,
                max_trials=50, convergence_enabled=False,
            )
            resumed_executor = RealCandidateExecutor(data_dir, state_dir, resumed.store, fingerprint)
            resumed.run(resumed_executor, target_trials=6)
            self.assertEqual(resumed.store.consumed, 6)
            self.assertEqual([item["optuna_trial_number"] for item in resumed.report()["trial_mapping"]], list(range(6)))

            inspected = inspect(data_dir, state_root)
            self.assertEqual(inspected["budget"]["used"], 6)
            self.assertEqual(len(inspected["trial_mapping"]), 6)

            # Terminal failures consume the same real-fingerprint budget. Fill the
            # isolated fixture ledger without spending compute, then prove that
            # both the loop and direct authority reject attempt 51.
            for ordinal in range(7, 51):
                failed = resumed.store.reserve(
                    "readiness_failure_fixture",
                    "terminal statuses count toward the hard cap",
                    {"fixture_ordinal": ordinal},
                )
                resumed.store.fail(failed["trial_id"], "intentional readiness fixture")
            self.assertEqual(resumed.store.consumed, 50)
            self.assertEqual(resumed.step(resumed_executor)["stop_reason"], "budget_exhausted")
            with self.assertRaises(BudgetExhausted):
                resumed.store.reserve("extra", "51st attempt", {})

            manifest = resumed.lock()
            self.assertTrue(manifest["immutable"])
            self.assertTrue(manifest["production_top1_eligible"])
            before_locked_step = resumed.store.consumed
            self.assertEqual(resumed.step(resumed_executor)["stop_reason"], "user_lock")
            self.assertEqual(resumed.store.consumed, before_locked_step)

            output = root / "locked-test-scores.csv"
            scoring = score_locked_exposures(data_dir, state_root, output, split="test")
            rows = load(str(data_dir))["test"]
            self.assertEqual(len(read_submission(output, rows)), len(rows))
            self.assertEqual(scoring["rows"], len(rows))
            self.assertFalse(scoring["metrics_computed"])
            self.assertFalse(scoring["test_metrics_used"])
            sidecar = json.loads(output.with_suffix(".csv.manifest.json").read_text())
            self.assertEqual(sidecar, scoring)

    def test_locked_history_uses_train_positives_and_ignores_validation_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            write_fixture_dataset(data_dir)
            splits = load(str(data_dir))
            enc, _ = encode(splits)
            rows = np.arange(len(splits["test"]), dtype=np.int64)
            before = frozen_train_positive_history_csr(
                data_dir, splits, enc, "test", rows, limit=None
            )
            mutated = {name: list(values) for name, values in splits.items()}
            mutated["valid"] = [tuple(list(row[:6]) + [1 - int(row[6])]) for row in splits["valid"]]
            after = frozen_train_positive_history_csr(
                data_dir, mutated, enc, "test", rows, limit=None
            )
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            # Two positive train events are repeated for every u1 test exposure.
            np.testing.assert_array_equal(np.diff(before[0]), np.full(len(rows), 2))


if __name__ == "__main__":
    unittest.main()
