import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import baseline as official_baseline

from tracks.agent_a.autonomous import AutonomousResearchLoop
from tracks.agent_a.behavioral_features import build_behavioral_features
from tracks.agent_a.candidate import (
    BackboneConfig, CandidateConfig, CandidateSpec, ListwiseConfig, RankerConfig,
)
from tracks.agent_a.finalize import score_locked_exposures
from tracks.agent_a.fingerprint import fingerprint_dataset
from tracks.agent_a.ranker import (
    normalize_within_user, predict_ranker_checkpoint, train_ranker_candidate,
)
from tracks.agent_a.safe_data import encode_research_splits, load_research_splits
from tracks.agent_a.tests.test_readiness import write_fixture_dataset


def ranker_config() -> CandidateConfig:
    return CandidateConfig(
        backbone=BackboneConfig("fm_lambdarank_ensemble", 16),
        ranker=RankerConfig(
            True, "causal_behavioral_v1", 20.0, 5,
            0.05, 7, 1, 1,
        ),
        listwise=ListwiseConfig(enabled=False),
    )


class BehavioralFeatureTest(unittest.TestCase):
    def test_train_aggregates_are_strictly_earlier_and_validation_is_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            write_fixture_dataset(data_dir)
            path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["video_id"] = rows[1]["video_id"] = rows[2]["video_id"] = "v1"
            rows[0]["time_ms"] = rows[1]["time_ms"] = "1000"
            rows[2]["time_ms"] = "1001"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            splits = load_research_splits(data_dir)
            enc, encoder = encode_research_splits(splits)
            before = build_behavioral_features(data_dir, enc, encoder)
            video_count_column = before.feature_names.index("video_count")
            self.assertEqual(before.train[0, video_count_column], 0.0)
            self.assertEqual(before.train[1, video_count_column], 0.0)
            self.assertAlmostEqual(before.train[2, video_count_column], np.log1p(2))

            validation_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
            with validation_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, later = reader.fieldnames, list(reader)
            for row in later:
                if 20220422 <= int(row["date"]) <= 20220428:
                    row["long_view"] = "0" if row["long_view"] != "0" else "1"
            with validation_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(later)
            mutated_splits = load_research_splits(data_dir)
            mutated_enc, mutated_encoder = encode_research_splits(mutated_splits)
            after = build_behavioral_features(data_dir, mutated_enc, mutated_encoder)
            np.testing.assert_array_equal(before.train, after.train)
            np.testing.assert_array_equal(before.evaluation, after.evaluation)

    def test_user_normalization_preserves_alignment_and_handles_constants(self):
        scores = normalize_within_user(np.asarray([1.0, 2.0, 5.0]), ["u1", "u1", "u2"])
        np.testing.assert_allclose(scores, [-1.0, 1.0, 0.0])


class RankerRoundtripTest(unittest.TestCase):
    def test_ranker_checkpoint_lock_and_label_free_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, state_root = root / "data", root / "state"
            write_fixture_dataset(data_dir)
            splits = load_research_splits(data_dir)
            enc, encoder = encode_research_splits(splits)
            fingerprint = fingerprint_dataset(data_dir)["dataset_fingerprint"]
            state_dir = state_root / fingerprint.removeprefix("sha256:")
            loop = AutonomousResearchLoop(
                state_dir, fingerprint, simulation=False,
                max_trials=50, convergence_enabled=False,
            )
            spec = CandidateSpec(fingerprint, ranker_config())
            trial = loop.store.reserve(
                "train_causal_behavioral_lambdarank_fm_ensemble",
                "fixture ranker roundtrip",
                spec.ledger_config(),
            )
            loop.store.mark_running(trial["trial_id"])
            model = official_baseline.FM(encoder.dimension, k=16, lr=0.001, seed=0)
            features = build_behavioral_features(data_dir, enc, encoder)
            _, outcome = train_ranker_candidate(
                enc, features, model, state_dir / "artifacts" / "ranker.npz",
                fingerprint, ensemble=True, n_estimators=5,
                learning_rate=0.05, num_leaves=7, min_child_samples=1,
                validation_interval=1,
            )
            completed = loop.store.complete(trial["trial_id"], outcome)
            artifact = completed["result"]["artifacts"][0]
            with np.load(artifact["path"], allow_pickle=False) as checkpoint:
                prediction = predict_ranker_checkpoint(
                    checkpoint, features.evaluation, enc["valid"][0], enc["valid"][2]
                )
            self.assertEqual(prediction.shape, (len(enc["valid"][1]),))
            self.assertTrue(np.all(np.isfinite(prediction)))
            loop.lock()
            output = root / "submission.csv"
            result = score_locked_exposures(data_dir, state_root, output, "test")
            self.assertEqual(result["rows"], 4)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
