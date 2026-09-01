import unittest

import numpy as np

from tracks.agent_a.listwise import ListwiseFM, group_user_exposures, listnet_loss_and_gradient


class ListwiseTest(unittest.TestCase):
    def test_grouping_is_same_user_discriminative_and_stable(self):
        users = ["u1", "u2", "u1", "u3", "u4", "u2", "u3", "u4", "u1"]
        labels = np.asarray([1, 0, 0, 1, 0, 0, 1, 1, 1], dtype=np.float64)
        groups = group_user_exposures(users, labels)
        self.assertEqual([group.user_id for group in groups], ["u1", "u4"])
        np.testing.assert_array_equal(groups[0].row_indices, [0, 2, 8])
        np.testing.assert_array_equal(groups[1].row_indices, [4, 7])
        for group in groups:
            self.assertEqual({users[index] for index in group.row_indices}, {group.user_id})
            self.assertGreater(group.positives, 0)
            self.assertLess(group.positives, len(group.row_indices))

    def test_hard_negatives_never_cross_users(self):
        users = ["a", "a", "a", "b", "b", "b"]
        labels = np.asarray([1, 0, 0, 0, 1, 0])
        scores = np.asarray([0.0, 0.1, 0.9, 0.8, 0.0, 0.2])
        groups = group_user_exposures(users, labels, scores, hard_negative_cap=1)
        np.testing.assert_array_equal(groups[0].row_indices, [0, 2])
        np.testing.assert_array_equal(groups[1].row_indices, [3, 4])

    def test_listnet_gradient_matches_finite_difference(self):
        logits = np.asarray([0.2, -0.4, 1.1, -0.3, 0.7], dtype=np.float64)
        labels = np.asarray([1, 0, 1, 0, 1], dtype=np.float64)
        sizes = [3, 2]
        loss, analytic = listnet_loss_and_gradient(
            logits, labels, sizes, score_temperature=0.8, target_temperature=0.5, group_weights=[2, 1]
        )
        numeric = np.empty_like(logits)
        epsilon = 1e-6
        for index in range(len(logits)):
            upper = logits.copy()
            lower = logits.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            upper_loss = listnet_loss_and_gradient(upper, labels, sizes, 0.8, 0.5, [2, 1])[0]
            lower_loss = listnet_loss_and_gradient(lower, labels, sizes, 0.8, 0.5, [2, 1])[0]
            numeric[index] = (upper_loss - lower_loss) / (2 * epsilon)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(analytic)))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(float(analytic[:3].sum()), 0.0, places=12)
        self.assertAlmostEqual(float(analytic[3:].sum()), 0.0, places=12)

    def test_extreme_logits_are_finite(self):
        loss, gradient = listnet_loss_and_gradient(
            np.asarray([1000.0, 999.0, -1000.0]), np.asarray([1, 0, 1]), [3]
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_baseline_initialization_copies_weights_and_resets_optimizer(self):
        class Baseline:
            V = np.arange(12, dtype=np.float32).reshape(6, 2)
            W = np.arange(6, dtype=np.float32)
            b = np.float32(0.25)

        model = ListwiseFM(Baseline(), lr=1e-6, weight_decay=0.0, warmup_steps=50)
        np.testing.assert_array_equal(model.V, Baseline.V)
        np.testing.assert_array_equal(model.W, Baseline.W)
        self.assertEqual(model.b, Baseline.b)
        self.assertEqual(model.t, 0)
        self.assertFalse(np.any(model.mV))
        self.assertFalse(np.any(model.vV))
        self.assertFalse(np.any(model.mW))
        self.assertFalse(np.any(model.vW))


if __name__ == "__main__":
    unittest.main()
