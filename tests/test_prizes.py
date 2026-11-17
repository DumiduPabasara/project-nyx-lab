import unittest

from prizes import (BASE_PRIZES, benchmark_odds, evaluate_prize,
                    expected_prize_per_ticket, matching_lengths,
                    prefix_suffix_distribution)


class PrizeTests(unittest.TestCase):
    def test_matching_lengths(self):
        self.assertEqual(matching_lengths('123456', '123456'), (6, 6))
        self.assertEqual(matching_lengths('123456', '123999'), (3, 0))
        self.assertEqual(matching_lengths('123456', '999456'), (0, 3))
        self.assertEqual(matching_lengths('123456', '923459'), (0, 0))
        with self.assertRaises(ValueError):
            matching_lengths('12345', '123456')

    def test_tiers(self):
        cases = [
            (('A', '123456', 'A', '123456'), 'super'),
            (('A', '123456', 'B', '123456'), 'six'),
            (('A', '123456', 'B', '123459'), 'first5'),
            (('A', '123456', 'B', '923456'), 'last5'),
            (('A', '123456', 'A', '923459'), 'letter'),
            (('A', '123456', 'B', '923459'), 'none'),
            (('A', '123456', 'B', '193456'), 'last4'),
        ]
        for args, tier in cases:
            self.assertEqual(evaluate_prize(*args).tier, tier, args)

    def test_highest_tier_wins_when_overlapping(self):
        # first1 (40) + last3 (2,000) + letter (40): only the largest pays.
        result = evaluate_prize('A', '123456', 'A', '199456')
        self.assertEqual((result.tier, result.amount), ('last3', 2000))

    def test_distribution_sums_to_one(self):
        self.assertAlmostEqual(sum(p for _, _, p in prefix_suffix_distribution()), 1.0, places=12)

    def test_expected_value_matches_enumeration_for_any_ticket(self):
        # Exact check over the whole 10^6 space for two very different tickets:
        # the analytic value equals brute force, and is the same for both.
        analytic = expected_prize_per_ticket()
        for ticket in ('123456', '000000'):
            hit = miss = 0
            for k in range(10**6):
                actual = f'{k:06d}'
                hit += evaluate_prize('A', ticket, 'A', actual).amount
                miss += evaluate_prize('A', ticket, 'B', actual).amount
            brute = (hit / 10**6) / 25 + (miss / 10**6) * 24 / 25
            self.assertAlmostEqual(analytic, brute, places=6, msg=ticket)

    def test_any_prize_probability_matches_tiers(self):
        odds = benchmark_odds()['any_prize_probability']
        self.assertAlmostEqual(odds, 1 - 0.96 * 0.81, places=12)

    def test_custom_table(self):
        table = dict(BASE_PRIZES, letter=0)
        self.assertEqual(evaluate_prize('A', '123456', 'A', '923459', prizes=table).amount, 0)


if __name__ == '__main__':
    unittest.main()