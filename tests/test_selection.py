import math
import random
import unittest

from model import (ALPHABET, MODELS, UNIFORM_LOG_LOSS, probability_vectors,
                   significantly_better, walkforward)


def random_history(n, seed):
    rng = random.Random(seed)
    return [dict(draw_number=i + 1, letter=rng.choice(ALPHABET),
                 digits=''.join(rng.choice('0123456789') for _ in range(6))) for i in range(n)]


class SelectionTests(unittest.TestCase):
    def test_uniform_loss_constant(self):
        self.assertAlmostEqual(UNIFORM_LOG_LOSS, (math.log(25) + 6 * math.log(10)) / 7)
        history = random_history(120, 1)
        _, scores = walkforward(history)
        self.assertAlmostEqual(scores['uniform']['validation_mean_log_loss'], UNIFORM_LOG_LOSS, places=9)

    def test_fast_scoring_matches_slow_reference(self):
        history = random_history(150, 3)
        _, scores = walkforward(history)
        evaluable = min(200, len(history) - 20)
        start = len(history) - evaluable
        mid = start + evaluable // 2
        for name in MODELS:
            losses = []
            for j in range(start, mid):
                vectors = probability_vectors(history[:j], name)
                actual = [ALPHABET.index(history[j]['letter'])] + [int(c) for c in history[j]['digits']]
                losses.append(sum(-math.log(max(v[a], 1e-15)) for v, a in zip(vectors, actual)) / 7)
            self.assertAlmostEqual(scores[name]['validation_mean_log_loss'], sum(losses) / len(losses), places=9)

    def test_random_data_stays_on_uniform(self):
        for seed in range(25):
            chosen, _ = walkforward(random_history(500, 100 + seed))
            self.assertEqual(chosen, 'uniform', seed)

    def test_strong_pattern_is_selected(self):
        rng = random.Random(7)
        # Heavily skewed letters and digits: a frequency model must clearly beat uniform.
        history = [dict(draw_number=i + 1,
                        letter='A' if rng.random() < 0.8 else rng.choice(ALPHABET),
                        digits=''.join('7' if rng.random() < 0.7 else rng.choice('0123456789') for _ in range(6)))
                   for i in range(500)]
        chosen, scores = walkforward(history)
        self.assertNotEqual(chosen, 'uniform')
        self.assertTrue(scores[chosen]['validation_beats_uniform'])

    def test_significance_helper(self):
        self.assertFalse(significantly_better(None, None, None))
        self.assertTrue(significantly_better(2.0, 1.99, 2.01))
        self.assertFalse(significantly_better(2.43, 2.30, 2.56))     # inside the noise
        self.assertFalse(significantly_better(UNIFORM_LOG_LOSS, 2.4, 2.5))


if __name__ == '__main__':
    unittest.main()