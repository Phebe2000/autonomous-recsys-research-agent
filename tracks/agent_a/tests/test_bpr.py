import unittest

import numpy as np

from tracks.agent_a.bpr import (
    BPRRegularizedListwiseFM,
    bpr_loss_and_gradient,
    listnet_bpr_loss_and_gradient,
)
from tracks.agent_a.listwise import ListwiseFM, listnet_loss_and_gradient


class BPRTest(unittest.TestCase):
    def test_bpr_gradient_matches_finite_difference_and_stays_in_groups(self):
        logits = np.asarray([0.3, -0.2, 0.8, 0.1, -0.4], dtype=np.float64)
        labels = np.asarray([1, 0, 1, 0, 1])
        sizes = [3, 2]
        loss, analytic = bpr_loss_and_gradient(logits, labels, sizes, [2, 1])
        numeric = np.empty_like(logits)
        epsilon = 1e-6
        for index in range(len(logits)):
            upper, lower = logits.copy(), logits.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            numeric[index] = (
                bpr_loss_and_gradient(upper, labels, sizes, [2, 1])[0]
                - bpr_loss_and_gradient(lower, labels, sizes, [2, 1])[0]
            ) / (2 * epsilon)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(float(analytic[:3].sum()), 0.0, places=12)
        self.assertAlmostEqual(float(analytic[3:].sum()), 0.0, places=12)

    def test_zero_weight_is_exact_listnet_parity(self):
        logits = np.asarray([0.2, -0.1, 0.5])
        labels = np.asarray([1, 0, 1])
        expected = listnet_loss_and_gradient(logits, labels, [3], 1.0, 0.5, [2])
        actual = listnet_bpr_loss_and_gradient(logits, labels, [3], 0.0, 1.0, 0.5, [2])
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(np.array_equal(expected[1], actual[1]))

    def test_zero_weight_model_step_matches_milestone1(self):
        class Baseline:
            V = np.arange(24, dtype=np.float32).reshape(12, 2) / 100
            W = np.arange(12, dtype=np.float32) / 100
            b = np.float32(0.2)

        reference = ListwiseFM(Baseline(), lr=1e-6, weight_decay=0.0, warmup_steps=0)
        candidate = BPRRegularizedListwiseFM(
            Baseline(), lr=1e-6, weight_decay=0.0, warmup_steps=0
        )
        X = np.asarray([[0, 5], [0, 6], [0, 7]], dtype=np.int32)
        y = np.asarray([1, 0, 1], dtype=np.float32)
        ref_loss = reference.step(X, y, [3], 1.0, 0.5, [2], True)
        bpr_loss = candidate.step(X, y, [3], 1.0, 0.5, [2], 0.0, True)
        self.assertEqual(ref_loss, bpr_loss)
        self.assertTrue(np.array_equal(reference.V, candidate.V))
        self.assertTrue(np.array_equal(reference.W, candidate.W))

    def test_extreme_pair_scores_are_finite(self):
        loss, gradient = bpr_loss_and_gradient(
            np.asarray([1000.0, -1000.0]), np.asarray([1, 0]), [2]
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(gradient)))


if __name__ == "__main__":
    unittest.main()
