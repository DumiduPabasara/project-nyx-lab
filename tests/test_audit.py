import math
import random
import unittest

from audit import (binomial_upper_tail, chi_square_sf, chi_square_uniform,
                   digit_hit_test, lag1_correlation, prize_hit_test,
                   randomness_report)
from model import ALPHABET


class AuditTests(unittest.TestCase):
    def test_chi_square_sf_known_values(self):
        self.assertAlmostEqual(chi_square_sf(3.841458820694124, 1), 0.05, places=6)
        self.assertAlmostEqual(chi_square_sf(16.918977604620448, 9), 0.05, places=6)
        self.assertEqual(chi_square_sf(0, 5), 1.0)

    def test_uniform_counts_pass_and_skewed_counts_fail(self):
        self.assertGreater(chi_square_uniform([100] * 10)[2], 0.99)
        self.assertLess(chi_square_uniform([300] + [77] * 9)[2], 1e-6)

    def test_binomial_tail(self):
        self.assertAlmostEqual(binomial_upper_tail(0, 10, 0.5), 1.0)
        self.assertAlmostEqual(binomial_upper_tail(10, 10, 0.5), 0.5 ** 10)
        self.assertAlmostEqual(binomial_upper_tail(6, 10, 0.5), 0.376953125, places=9)
        self.assertEqual(binomial_upper_tail(11, 10, 0.5), 0.0)

    def test_hit_tests(self):
        result = prize_hit_test(1, 9, 0.2224)
        self.assertAlmostEqual(result['expected'], 9 * 0.2224)
        self.assertGreater(result['p_at_least'], 0.8)   # 1 of 9 is unremarkable
        self.assertAlmostEqual(digit_hit_test(3, 5)['expected'], 3.0)

    def test_lag1_detects_dependence(self):
        rng = random.Random(1)
        independent = [rng.randrange(10) for _ in range(2000)]
        self.assertGreater(lag1_correlation(independent)[1], 0.01)
        sticky = [0]
        for _ in range(2000):
            sticky.append(sticky[-1] if rng.random() < 0.7 else rng.randrange(10))
        self.assertLess(lag1_correlation(sticky)[1], 1e-6)

    def test_report_structure(self):
        rng = random.Random(2)
        history = [dict(letter=rng.choice(ALPHABET), digits=''.join(rng.choice('0123456789') for _ in range(6)))
                   for _ in range(500)]
        report = randomness_report(history, ALPHABET)
        self.assertEqual(report['n_tests'], 13)
        self.assertAlmostEqual(report['bonferroni_threshold'], 0.05 / 13)
        self.assertTrue(all(0 <= t['p_value'] <= 1 for t in report['tests']))


if __name__ == '__main__':
    unittest.main()