import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tracks.agent_a.history import (
    CausalPositiveHistory,
    build_causal_history,
    csr_mean_pool,
)
from tracks.agent_a.history_model import (
    HistoryListwiseFM,
    history_residual_and_gradients,
    load_history_checkpoint,
    save_history_checkpoint,
)


def official_row(date, user, video, label):
    return (date, user, video, "author", "tab", 1000.0, label)


class HistoryBuilderTest(unittest.TestCase):
    def fixture(self, root: Path, valid_labels=(1, 0)):
        root.mkdir()
        header = ["date", "user_id", "video_id", "time_ms"]
        train = [
            [20220410, "u", "v3", 30],
            [20220410, "u", "v1", 10],
            [20220410, "u", "v2", 20],
            [20220410, "u", "v4", 20],
        ]
        valid = [[20220422, "u", "v5", 40], [20220422, "cold", "v6", 40]]
        for name, rows in (
            ("log_standard_4_08_to_4_21_pure.csv", train),
            ("log_standard_4_22_to_5_08_pure.csv", valid),
        ):
            with (root / name).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
        splits = {
            "train": [
                official_row(20220410, "u", "v3", 0),
                official_row(20220410, "u", "v1", 1),
                official_row(20220410, "u", "v2", 1),
                official_row(20220410, "u", "v4", 0),
            ],
            "valid": [
                official_row(20220422, "u", "v5", valid_labels[0]),
                official_row(20220422, "cold", "v6", valid_labels[1]),
            ],
        }
        Xtr = np.asarray([[0, 13], [0, 11], [0, 12], [0, 14]], dtype=np.int32)
        Xva = np.asarray([[0, 15], [1, 16]], dtype=np.int32)
        enc = {
            "train": (Xtr, np.asarray([0, 1, 1, 0], dtype=np.float32), ["u"] * 4),
            "valid": (Xva, np.asarray(valid_labels, dtype=np.float32), ["u", "cold"]),
        }
        return splits, enc

    def test_strictly_earlier_excludes_current_future_and_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            splits, enc = self.fixture(Path(directory) / "data")
            history = build_causal_history(Path(directory) / "data", splits, enc)
            expected = ([11, 12], [], [11], [11])
            for row, items in enumerate(expected):
                _, actual = history.batch_csr("train", [row], None)
                np.testing.assert_array_equal(actual, items)

    def test_validation_is_frozen_train_only_and_cold_start(self):
        with tempfile.TemporaryDirectory() as directory:
            first_root = Path(directory) / "first"
            second_root = Path(directory) / "second"
            splits_a, enc_a = self.fixture(first_root, (1, 0))
            splits_b, enc_b = self.fixture(second_root, (0, 1))
            first = build_causal_history(first_root, splits_a, enc_a)
            second = build_causal_history(second_root, splits_b, enc_b)
            ptr_a, ids_a = first.batch_csr("valid", [0, 1], None)
            ptr_b, ids_b = second.batch_csr("valid", [0, 1], None)
            np.testing.assert_array_equal(ptr_a, [0, 2, 2])
            np.testing.assert_array_equal(ids_a, [11, 12])
            np.testing.assert_array_equal(ptr_a, ptr_b)
            np.testing.assert_array_equal(ids_a, ids_b)
            with self.assertRaises(ValueError):
                first.batch_csr("test", [0], None)

    def test_last_n_all_and_mean_pooling(self):
        items = np.arange(1, 121, dtype=np.int32)
        history = CausalPositiveHistory(
            (np.arange(120, dtype=np.int64),),
            (items,),
            np.asarray([0]),
            np.asarray([120]),
            np.asarray([0]),
        )
        for limit in (20, 50, 100, None):
            _, actual = history.batch_csr("train", [0], limit)
            size = 120 if limit is None else limit
            np.testing.assert_array_equal(actual, items[-size:])
        embeddings = np.arange(12, dtype=np.float64).reshape(6, 2)
        pooled = csr_mean_pool(np.asarray([0, 0, 3]), np.asarray([1, 2, 2]), embeddings)
        np.testing.assert_array_equal(pooled[0], [0.0, 0.0])
        np.testing.assert_allclose(pooled[1], (embeddings[1] + 2 * embeddings[2]) / 3)


class HistoryModelTest(unittest.TestCase):
    class Baseline:
        V = np.arange(24, dtype=np.float32).reshape(12, 2) / 100
        W = np.arange(12, dtype=np.float32) / 100
        b = np.float32(0.1)

    def test_gate_zero_is_bitwise_fm_parity_and_score_length_guard(self):
        model = HistoryListwiseFM(self.Baseline(), 1e-6, 0.0, 0, history_dim=2, gate=0.0)
        X = np.asarray([[0, 5], [1, 6]], dtype=np.int32)
        ptr = np.asarray([0, 1, 1])
        ids = np.asarray([4])
        expected = model.fm_forward(X)[0]
        actual = model.scores(X, ptr, ids)
        self.assertTrue(np.array_equal(expected, actual))
        with self.assertRaises(ValueError):
            model.scores(X, np.asarray([0, 1]), ids)

    def test_history_embedding_and_gate_gradients_match_finite_difference(self):
        H = np.asarray([[0.2, -0.1], [0.3, 0.5], [-0.4, 0.7]], dtype=np.float64)
        candidates = np.asarray([0, 1])
        ptr = np.asarray([0, 2, 3])
        ids = np.asarray([1, 2, 0])
        upstream = np.asarray([0.7, -0.2])
        gate = 0.8
        residual, analytic_h, analytic_gate = history_residual_and_gradients(
            candidates, ptr, ids, H, gate, upstream
        )

        def objective(table, scalar):
            values = history_residual_and_gradients(candidates, ptr, ids, table, scalar)[0]
            return float(np.dot(upstream, values))

        numeric_h = np.empty_like(H)
        epsilon = 1e-6
        for index in np.ndindex(H.shape):
            upper, lower = H.copy(), H.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            numeric_h[index] = (objective(upper, gate) - objective(lower, gate)) / (2 * epsilon)
        numeric_gate = (objective(H, gate + epsilon) - objective(H, gate - epsilon)) / (2 * epsilon)
        self.assertEqual(len(residual), len(candidates))
        np.testing.assert_allclose(analytic_h, numeric_h, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(analytic_gate, numeric_gate, places=7)

    def test_checkpoint_roundtrip_includes_optimizer_gate_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            model = HistoryListwiseFM(self.Baseline(), 1e-6, 0.0, 50, history_dim=2, gate=1.0)
            model.H += 0.3
            model.mH += 0.2
            model.vH += 0.4
            model.mg = np.float32(0.5)
            model.vg = np.float32(0.6)
            model.t = 7
            config = {"run_id": "H-01", "history": {"last_n": 20}}
            path = Path(directory) / "checkpoint.npz"
            save_history_checkpoint(path, model, "sha256:data", "trial-05", config, {"step": 7})
            restored = HistoryListwiseFM(self.Baseline(), 1e-6, 0.0, 50, history_dim=2, gate=0.0)
            progress = load_history_checkpoint(
                path, restored, "sha256:data", "trial-05", config
            )
            self.assertEqual(progress, {"step": 7})
            for key, value in model.snapshot().items():
                np.testing.assert_array_equal(restored.snapshot()[key], value)
            with self.assertRaises(ValueError):
                load_history_checkpoint(path, restored, "sha256:wrong", "trial-05", config)

    def test_gate_only_freeze_and_delayed_history_unfreeze(self):
        X = np.asarray([[0, 5], [0, 6], [0, 7]], dtype=np.int32)
        labels = np.asarray([1, 0, 1], dtype=np.float32)
        ptr = np.asarray([0, 1, 2, 3])
        ids = np.asarray([1, 2, 3])
        frozen = HistoryListwiseFM(
            self.Baseline(),
            1e-6,
            0.0,
            0,
            history_dim=2,
            gate=0.0,
            gate_lr=1e-2,
            train_history_embeddings=False,
        )
        initial_h = frozen.H.copy()
        frozen.step(X, labels, [3], ptr, ids, 1.0, 0.5, [2], True)
        self.assertTrue(np.array_equal(initial_h, frozen.H))
        self.assertNotEqual(float(frozen.gate), 0.0)

        delayed = HistoryListwiseFM(
            self.Baseline(),
            1e-6,
            0.0,
            0,
            history_dim=2,
            gate=0.0,
            history_lr=1e-2,
            gate_lr=1e-2,
            history_unfreeze_step=1,
        )
        initial_h = delayed.H.copy()
        delayed.step(X, labels, [3], ptr, ids, 1.0, 0.5, [2], True)
        self.assertTrue(np.array_equal(initial_h, delayed.H))
        delayed.step(X, labels, [3], ptr, ids, 1.0, 0.5, [2], True)
        self.assertFalse(np.array_equal(initial_h, delayed.H))
        self.assertEqual(delayed.tH, 1)

    def test_fixed_gate_and_frozen_history_remain_exactly_constant(self):
        X = np.asarray([[0, 5], [0, 6], [0, 7]], dtype=np.int32)
        labels = np.asarray([1, 0, 1], dtype=np.float32)
        ptr = np.asarray([0, 1, 2, 3])
        ids = np.asarray([1, 2, 3])
        model = HistoryListwiseFM(
            self.Baseline(),
            1e-3,
            0.0,
            0,
            history_dim=2,
            gate=-0.05,
            train_history_embeddings=False,
            train_gate=False,
        )
        initial_h = model.H.copy()
        initial_gate = model.gate.copy()
        initial_v = model.V.copy()
        model.step(X, labels, [3], ptr, ids, 1.0, 0.5, [2], True)
        self.assertTrue(np.array_equal(initial_h, model.H))
        self.assertTrue(np.array_equal(initial_gate, model.gate))
        self.assertEqual(model.tH, 0)
        self.assertEqual(float(model.mg), 0.0)
        self.assertEqual(float(model.vg), 0.0)
        self.assertFalse(np.array_equal(initial_v, model.V))


if __name__ == "__main__":
    unittest.main()
