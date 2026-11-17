import unittest

from prizes import (BASE_PRIZES, SUPER_PRIZE_REFERENCE, TICKET_PRICE, expected_prize_by_tier, return_to_player, benchmark_odds, evaluate_prize,
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

    def test_prize_table_matches_official_structure(self):
        official = {'super': 22_718_328, 'six': 2_500_000, 'last5': 100_000, 'last4': 15_000,
                    'last3': 2_000, 'last2': 200, 'last1': 40, 'first5': 100_000,
                    'first4': 2_000, 'first3': 200, 'first2': 80, 'first1': 40, 'letter': 40}
        self.assertEqual(BASE_PRIZES, official)
        self.assertEqual(SUPER_PRIZE_REFERENCE, official['super'])

    def test_equal_amount_ties_resolve_last_then_first_then_letter(self):
        # Verified against the official winner counts of the 17 Nov 2026 draw.
        # Ticket 123456. The only reachable equal-amount ties are
        # {last1, first1, letter} (Rs 40) and {last2, first3} (Rs 200).
        self.assertEqual(evaluate_prize('A', '123456', 'A', '199996').tier, 'last1')   # first1+last1+letter
        self.assertEqual(evaluate_prize('A', '123456', 'B', '199996').tier, 'last1')   # first1+last1
        self.assertEqual(evaluate_prize('A', '123456', 'A', '199999').tier, 'first1')  # first1+letter
        self.assertEqual(evaluate_prize('A', '123456', 'B', '123956').tier, 'last2')   # first3+last2
        # Non-ties: the larger amount always wins.
        self.assertEqual(evaluate_prize('A', '123456', 'B', '129999').tier, 'first2')
        self.assertEqual(evaluate_prize('A', '123456', 'B', '123496').tier, 'first4')
        self.assertEqual(evaluate_prize('A', '123456', 'B', '129456').tier, 'last3')
        self.assertEqual(evaluate_prize('A', '123456', 'B', '923456').tier, 'last5')

    def test_return_to_player_and_tier_breakdown(self):
        self.assertEqual(TICKET_PRICE, 40)
        self.assertAlmostEqual(return_to_player(), expected_prize_per_ticket() / 40)
        self.assertAlmostEqual(sum(expected_prize_by_tier().values()), expected_prize_per_ticket(), places=9)
        with self.assertRaises(ValueError):
            return_to_player(ticket_price=0)

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