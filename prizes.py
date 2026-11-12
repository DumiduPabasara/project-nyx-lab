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


def evaluate_prize(predicted_letter, predicted_digits, actual_letter, actual_digits, prizes=None):
    """Score one recorded forecast. Highest payable tier wins when tiers overlap."""
    table = BASE_PRIZES if prizes is None else prizes
    prefix, suffix = matching_lengths(predicted_digits, actual_digits)
    letter_hit = predicted_letter == actual_letter
    eligible = []
    if prefix == 6:
        eligible.append('super' if letter_hit else 'six')
    else:
        eligible.extend(f'first{i}' for i in range(1, min(prefix, 5) + 1))
        eligible.extend(f'last{i}' for i in range(1, min(suffix, 5) + 1))
        if letter_hit:
            eligible.append('letter')
    tier = max(eligible, key=lambda name: table[name], default='none')
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
