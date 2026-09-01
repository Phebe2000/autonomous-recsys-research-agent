import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tracks.agent_a.auxiliary import (
    PlayTimeTransform,
    RawIdentity,
    binary_cross_entropy_with_gradient,
    huber_with_gradient,
    load_training_auxiliary,
    read_training_identities,
)
from tracks.agent_a.listwise import ListwiseFM
from tracks.agent_a.multitask_model import (
    MultiTaskListwiseFM,
    load_multitask_checkpoint,
    save_multitask_checkpoint,
)
from tracks.agent_a.contracts import ContractError, ValidationMetrics
from tracks.agent_a.store import ResearchStore
from submit import HEADER


class Baseline:
    V = np.arange(24, dtype=np.float32).reshape(12, 2) / 100
    W = np.arange(12, dtype=np.float32) / 100
    b = np.float32(0.1)


class AuxiliaryAdapterTest(unittest.TestCase):
    def fixture(self, root: Path):
        root.mkdir()
        header = [
            "user_id", "video_id", "date", "time_ms", "is_click", "long_view",
            "play_time_ms",
        ]
        rows = [
            ["u1", "v1", 20220410, 100, 1, 1, 999],
            ["u1", "v2", 20220410, 101, 0, 0, 10],
        ]
        with (root / "log_standard_4_08_to_4_21_pure.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        splits = {
            "train": [
                (20220410, "u1", "v1", "a", "t", 1.0, 1),
                (20220410, "u1", "v2", "a", "t", 1.0, 0),
            ]
        }
        enc = {
            "train": (
                np.asarray([[0, 2], [0, 3]], dtype=np.int32),
                np.asarray([1, 0], dtype=np.float32),
                ["u1", "u1"],
            )
        }
        return splits, enc

    def test_exact_alignment_and_train_only_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            splits, enc = self.fixture(root)
            identities = read_training_identities(root)
            labels = load_training_auxiliary(root, splits, enc, identities)
            np.testing.assert_array_equal(labels.is_click, [1, 0])
            self.assertEqual(identities[0], RawIdentity(20220410, "u1", "v1", 100))
            labels.for_split("train")
            for split in ("valid", "test"):
                with self.assertRaises(ValueError):
                    labels.for_split(split)

    def test_misaligned_identity_and_missing_row_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            splits, enc = self.fixture(root)
            identities = list(read_training_identities(root))
            identities[1] = RawIdentity(20220410, "u1", "v2", 999)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_training_auxiliary(root, splits, enc, identities)
            enc["train"] = (enc["train"][0][:-1], enc["train"][1][:-1], enc["train"][2][:-1])
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                load_training_auxiliary(root, splits, enc)


class ObjectiveTest(unittest.TestCase):
    @staticmethod
    def finite_difference(function, values):
        epsilon = 1e-6
        result = np.empty_like(values, dtype=np.float64)
        for index in range(len(values)):
            upper, lower = values.copy(), values.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            result[index] = (function(upper) - function(lower)) / (2 * epsilon)
        return result

    def test_bce_gradient_and_extreme_logits(self):
        logits = np.asarray([-0.7, 0.2, 1.4])
        targets = np.asarray([0, 1, 1])
        loss, gradient = binary_cross_entropy_with_gradient(logits, targets)
        numeric = self.finite_difference(
            lambda value: binary_cross_entropy_with_gradient(value, targets)[0], logits
        )
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-7)
        extreme_loss, extreme_gradient = binary_cross_entropy_with_gradient(
            np.asarray([-1000.0, 1000.0]), np.asarray([0.0, 1.0])
        )
        self.assertTrue(np.isfinite(loss) and np.isfinite(extreme_loss))
        self.assertTrue(np.all(np.isfinite(extreme_gradient)))

    def test_huber_gradient_and_train_fitted_transform(self):
        prediction = np.asarray([-2.0, 0.2, 2.0])
        target = np.asarray([0.0, 0.0, 0.0])
        _, gradient = huber_with_gradient(prediction, target, delta=1.0)
        numeric = self.finite_difference(
            lambda value: huber_with_gradient(value, target, 1.0)[0], prediction
        )
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-7)
        transform = PlayTimeTransform.fit_train([0, 9, 99])
        expected = transform.transform([0, 9, 99])
        self.assertAlmostEqual(float(expected.mean()), 0.0, places=6)
        # Transforming another split cannot mutate train statistics.
        before = transform.to_dict()
        transform.transform([999999])
        self.assertEqual(before, transform.to_dict())


class MultiTaskModelTest(unittest.TestCase):
    def batch(self):
        return (
            np.asarray([[0, 5], [0, 6], [0, 7]], dtype=np.int32),
            np.asarray([1, 0, 1], dtype=np.float32),
            np.asarray([1, 0, 1], dtype=np.float32),
            np.asarray([0.5, -0.3, 1.2], dtype=np.float32),
        )

    def test_zero_weight_exact_loss_gradient_and_update_parity(self):
        X, y, click, play = self.batch()
        reference = ListwiseFM(Baseline(), 1e-3, 0.0, 0)
        model = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.0, 0.0)
        expected_loss = reference.step(X, y, [3], 1.0, 0.5, [2], True)
        losses = model.step(X, y, click, play, [3], 1.0, 0.5, [2], True)
        self.assertEqual(losses["total"], expected_loss)
        for name in ("V", "W", "mV", "vV", "mW", "vW"):
            self.assertTrue(np.array_equal(getattr(model.core, name), getattr(reference, name)))
        self.assertEqual(model.core.t, reference.t)

    def test_auxiliary_gradient_reaches_shared_embedding_and_heads_are_isolated(self):
        X, y, click, play = self.batch()
        reference = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.0, 0.0)
        click_model = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.05, 0.0)
        click_model.click_head[:] = [0.2, -0.1]
        reference.click_head[:] = click_model.click_head
        reference.step(X, y, click, play, [3], 1.0, 0.5, [2], True)
        click_model.step(X, y, click, play, [3], 1.0, 0.5, [2], True)
        self.assertFalse(np.array_equal(reference.V, click_model.V))
        self.assertFalse(np.all(click_model.click_head == np.asarray([0.2, -0.1])))
        self.assertTrue(np.all(click_model.play_head == 0))

    def test_checkpoint_roundtrip_heads_optimizer_config_and_normalization(self):
        X, y, click, play = self.batch()
        config = {"weights": {"is_click": 0.01, "play_time": 0.01}}
        normalization = {"play_time": {"mean": 1.2, "scale": 0.4}}
        with tempfile.TemporaryDirectory() as directory:
            model = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.01, 0.01)
            model.step(X, y, click, play, [3], 1.0, 0.5, [2], True)
            path = Path(directory) / "model.npz"
            save_multitask_checkpoint(
                path, model, "sha256:data", "trial-21", config, normalization, {"step": 1}
            )
            restored = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.01, 0.01)
            progress = load_multitask_checkpoint(
                path, restored, "sha256:data", "trial-21", config, normalization
            )
            self.assertEqual(progress, {"step": 1})
            for key, value in model.snapshot().items():
                np.testing.assert_array_equal(restored.snapshot()[key], value)
            with self.assertRaises(ValueError):
                load_multitask_checkpoint(
                    path, restored, "sha256:data", "trial-21", config,
                    {"play_time": {"mean": 0, "scale": 1}},
                )

    def test_resume_preserves_budget_and_multitask_optimizer_state(self):
        X, y, click, play = self.batch()
        config = {"weights": {"is_click": 0.01, "play_time": 0.0}}
        normalization = {"play_time": {"mean": 1.2, "scale": 0.4}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResearchStore(root / "ledger.sqlite3", "sha256:data")
            trial = store.reserve("multitask", "resume state", config)
            model = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.01, 0.0)
            model.step(X, y, click, play, [3], 1.0, 0.5, [2], True)
            save_multitask_checkpoint(
                root / "latest.npz", model, "sha256:data", trial["trial_id"],
                config, normalization, {"step": 1},
            )
            reopened = ResearchStore(root / "ledger.sqlite3", "sha256:data")
            resumed = reopened.resume(trial["trial_id"])
            self.assertEqual(reopened.consumed, 1)
            restored = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.01, 0.0)
            progress = load_multitask_checkpoint(
                root / "latest.npz", restored, "sha256:data", resumed["trial_id"],
                config, normalization,
            )
            self.assertEqual(progress["step"], 1)
            self.assertEqual(restored.core.t, 1)
            np.testing.assert_array_equal(restored.click_head, model.click_head)

    def test_one_long_view_score_per_row(self):
        X, _, _, _ = self.batch()
        model = MultiTaskListwiseFM(Baseline(), 1e-3, 0.0, 0, 0.01, 0.01)
        scores = model.predict(X)
        self.assertEqual(scores.ndim, 1)
        self.assertEqual(len(scores), len(X))
        self.assertEqual(HEADER, ["row_id", "user_id", "video_id", "score"])

    def test_auxiliary_validation_metrics_are_rejected_for_selection(self):
        with self.assertRaises(ContractError):
            ValidationMetrics.from_mapping(
                {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55, "is_click": 0.9}
            )


if __name__ == "__main__":
    unittest.main()
