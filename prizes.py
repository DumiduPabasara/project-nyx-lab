"""prize-aware evaluation.
"""

from dataclasses import dataclass

BASE_PRIZES = {
    'super': 21_167_456,
    'six': 2_500_000,
    'last5': 100_000,
    'last4': 15_000,
    'last3': 2_000,
    'last2': 200,
    'last1': 40,
    'first5': 100_000,
    'first4': 2_000,
    'first3': 200,
    'first2': 80,
    'first1': 40,
    'letter': 40,
}

@dataclass(frozen=True)
class PrizeResult:
    tier: str
    amount: int
    prefix: int
    suffix: int
    letter_hit: bool


def matching_lengths(predicted_digits, actual_digits):
    if not (isinstance(predicted_digits, str) and isinstance(actual_digits, str)
            and len(predicted_digits) == len(actual_digits) == 6
            and predicted_digits.isdigit() and actual_digits.isdigit()):
        raise ValueError('Both numbers must be six-character digit strings')
    prefix = 0
    for a, b in zip(predicted_digits, actual_digits):
        if a != b:
            break
        prefix += 1
    suffix = 0
    for a, b in zip(reversed(predicted_digits), reversed(actual_digits)):
        if a != b:
            break
        suffix += 1
    return prefix, suffix


def best_tier(prefix, suffix, letter_hit, table=None):
    """Highest payable tier for a (prefix, suffix, letter) outcome."""
    table = BASE_PRIZES if table is None else table
    eligible = []
    if prefix == 6:
        eligible.append('super' if letter_hit else 'six')
    else:
        eligible.extend(f'first{i}' for i in range(1, min(prefix, 5) + 1))
        eligible.extend(f'last{i}' for i in range(1, min(suffix, 5) + 1))
        if letter_hit:
            eligible.append('letter')
    return max(eligible, key=lambda name: table[name], default='none')


def evaluate_prize(predicted_letter, predicted_digits, actual_letter, actual_digits, prizes=None):
    """Score one recorded forecast. Highest payable tier wins when tiers overlap.

    NOTE: "highest tier only" is an assumption; confirm it against the
    official prize rules (some lotteries pay overlapping tiers cumulatively).
    """
    table = BASE_PRIZES if prizes is None else prizes
    prefix, suffix = matching_lengths(predicted_digits, actual_digits)
    letter_hit = predicted_letter == actual_letter
    tier = best_tier(prefix, suffix, letter_hit, table)
    return PrizeResult(tier, table[tier] if tier != 'none' else 0,
                       prefix, suffix, letter_hit)


def benchmark_odds(letter_count=25):
    """Exact theoretical chances per ticket under independent, uniform draws.

    Prefix/suffix events below are inclusive, not mutually exclusive payout tiers.
    """
    if letter_count < 1:
        raise ValueError('letter_count must be positive')
    return {
        'super_exact_probability': 1 / (letter_count * 10**6),
        'six_digits_probability': 1 / 10**6,
        'five_digit_prefix_probability': 1 / 10**5,
        'five_digit_suffix_probability': 1 / 10**5,
        'letter_probability': 1 / letter_count,
        'any_prize_probability': 1 - (1 - 1 / letter_count) * (9 / 10)**2,
    }


def prefix_suffix_distribution():
    """Exact joint law of (prefix, suffix) match lengths for one random 6-digit draw.

    The probabilities are identical for EVERY ticket number, which is why no
    choice of number can change a ticket's expected prize.
    """
    yield 6, 6, 10.0 ** -6
    for prefix in range(6):
        for suffix in range(6 - prefix):          # prefix + suffix <= 5
            matched = prefix + suffix
            mismatches = 1 if matched == 5 else 2  # first and last mismatch may coincide
            yield prefix, suffix, 0.1 ** matched * 0.9 ** mismatches


def expected_prize_per_ticket(letter_count=25, prizes=None):
    """Exact expected payout of one ticket (any number) under the prize table."""
    table = BASE_PRIZES if prizes is None else prizes
    p_letter = 1 / letter_count
    total = 0.0
    for prefix, suffix, probability in prefix_suffix_distribution():
        hit_tier = best_tier(prefix, suffix, True, table)
        miss_tier = best_tier(prefix, suffix, False, table)
        hit = table[hit_tier] if hit_tier != 'none' else 0
        miss = table[miss_tier] if miss_tier != 'none' else 0
        total += probability * (p_letter * hit + (1 - p_letter) * miss)
    return total