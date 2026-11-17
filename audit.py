"""Randomness and luck-vs-skill statistics (pure Python, no SciPy needed).

These tests can only fail to reject "the draws are random". A small p-value
among many tests is expected by chance, so every report includes the number of
tests run and a Bonferroni-corrected threshold.
"""
import math

DIGITS = 6


def chi_square_sf(x, df):
    """P(Chi2_df >= x), via the regularised upper incomplete gamma function."""
    if x <= 0:
        return 1.0
    a, y = df / 2.0, x / 2.0
    log_prefactor = -y + a * math.log(y) - math.lgamma(a)
    if y < a + 1:  # series for the lower function
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= y / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(log_prefactor)))
    tiny = 1e-300  # continued fraction for the upper function (Lentz)
    b = y + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return max(0.0, min(1.0, math.exp(log_prefactor) * h))


def chi_square_uniform(counts):
    """Chi-square goodness of fit against equal expected counts."""
    total = sum(counts)
    expected = total / len(counts)
    statistic = sum((c - expected) ** 2 / expected for c in counts)
    df = len(counts) - 1
    return statistic, df, chi_square_sf(statistic, df)


def lag1_correlation(values):
    """Pearson r between consecutive draws, with a normal-approximation p-value."""
    n = len(values) - 1
    if n < 10:
        return None, None
    x, y = values[:-1], values[1:]
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None, None
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, math.erfc(abs(t) / math.sqrt(2))


def binomial_upper_tail(k, n, p):
    """Exact P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        log_pmf = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                   + i * log_p + (n - i) * log_q)
        total += math.exp(log_pmf)
    return min(1.0, total)


def prize_hit_test(hits, forecasts, chance_probability):
    """How surprising is this many prize hits if every forecast were a random ticket?"""
    expected = forecasts * chance_probability
    return {'observed': hits, 'expected': expected,
            'p_at_least': binomial_upper_tail(hits, forecasts, chance_probability)}


def digit_hit_test(total_position_hits, forecasts):
    """Same idea for correct digit positions (6 per forecast, 10% chance each)."""
    trials = forecasts * DIGITS
    return {'observed': total_position_hits, 'expected': trials * 0.1,
            'p_at_least': binomial_upper_tail(total_position_hits, trials, 0.1)}


def randomness_report(history, letters):
    """Run the standard battery on a chronological list of draw dicts."""
    rows = []
    digit_rows = [[int(c) for c in r['digits']] for r in history]
    for pos in range(DIGITS):
        counts = [0] * 10
        for digits in digit_rows:
            counts[digits[pos]] += 1
        stat, df, p = chi_square_uniform(counts)
        rows.append({'test': f'Digit position {pos + 1}: uniformity (chi-square)',
                     'statistic': stat, 'p_value': p})
    letter_counts = [0] * len(letters)
    for r in history:
        letter_counts[letters.index(r['letter'])] += 1
    stat, df, p = chi_square_uniform(letter_counts)
    rows.append({'test': 'Letter: uniformity (chi-square)', 'statistic': stat, 'p_value': p})
    for pos in range(DIGITS):
        r, p = lag1_correlation([digits[pos] for digits in digit_rows])
        if r is not None:
            rows.append({'test': f'Digit position {pos + 1}: lag-1 correlation',
                         'statistic': r, 'p_value': p})
    tests = len(rows)
    threshold = 0.05 / tests if tests else None
    for row in rows:
        row['flagged'] = threshold is not None and row['p_value'] < threshold
    return {'tests': rows, 'n_tests': tests, 'bonferroni_threshold': threshold,
            'draws': len(history)}