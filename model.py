"""Walk-forward statistical baselines; no claim of lottery predictability.

Selection rule: a non-uniform model may replace ``uniform`` only if its mean
validation log loss beats the uniform loss by more than ``SIGNIFICANCE_Z``
standard errors. With ~100 validation cases and several candidates, the
nominal "best" model is usually just the luckiest one, so uniform is the
default unless the evidence is strong. The holdout window is reporting-only.
"""
import math
from collections import Counter

ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MODELS = ("uniform", "all_history", "recent_30", "recent_100")
WINDOWS = {"recent_30": 30, "recent_100": 100}
SYMBOL_SIZES = (len(ALPHABET), 10, 10, 10, 10, 10, 10)
SMOOTHING_STRENGTH = 20.0
# Mean log loss (per symbol) of a perfectly uniform forecaster. It is the same
# for every draw, so any model's loss is already a *paired* difference vs. it.
UNIFORM_LOG_LOSS = sum(math.log(size) for size in SYMBOL_SIZES) / len(SYMBOL_SIZES)
# Conservative: ~99% one-sided, which also absorbs the several candidates tried.
SIGNIFICANCE_Z = 2.58


def observed_probs(rows, model, alphabet_size):
    if model == 'uniform' or not rows:
        return [1 / alphabet_size] * alphabet_size
    if model == 'recent_30': rows = rows[-30:]
    elif model == 'recent_100': rows = rows[-100:]
    # Dirichlet smoothing means limited data don't produce false certainty.
    counts = Counter(rows)
    strength = SMOOTHING_STRENGTH
    return [(counts.get(i, 0) + strength / alphabet_size) / (len(rows) + strength) for i in range(alphabet_size)]


def probability_vectors(history, model):
    letters = [ALPHABET.index(r['letter']) for r in history]
    vecs = [observed_probs(letters, model, len(ALPHABET))]
    for pos in range(6):
        vecs.append(observed_probs([int(r['digits'][pos]) for r in history], model, 10))
    return vecs


def _mean_95_interval(values):
    if len(values) < 2:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return mean - margin, mean + margin


def significantly_better(mean, ci_low, ci_high, z=SIGNIFICANCE_Z, baseline=UNIFORM_LOG_LOSS):
    """True only if ``mean`` is below ``baseline`` by more than ``z`` standard errors.

    The standard error is recovered from the stored 95% interval, so this works
    for classical and neural score summaries alike.
    """
    if mean is None or ci_low is None or ci_high is None:
        return False
    standard_error = (ci_high - ci_low) / (2 * 1.96)
    return mean + z * standard_error < baseline


# --- fast incremental scoring -------------------------------------------------
# Prefix counts let every walk-forward step read its window in O(alphabet)
# instead of re-counting the whole history (was O(n) per step).

def _labels(history):
    return [[ALPHABET.index(r['letter'])] + [int(c) for c in r['digits']] for r in history]


def _prefix_counts(labels):
    tables = []
    for pos, size in enumerate(SYMBOL_SIZES):
        running = [0] * size
        table = [tuple(running)]
        for label in labels:
            running[label[pos]] += 1
            table.append(tuple(running))
        tables.append(table)
    return tables


def _vectors_at(prefix, model, end):
    """Probability vectors for a forecast made with draws [0, end) known."""
    if model == 'uniform' or end == 0:
        return [[1 / size] * size for size in SYMBOL_SIZES]
    start = 0 if model == 'all_history' else max(0, end - WINDOWS[model])
    window = end - start
    strength = SMOOTHING_STRENGTH
    vectors = []
    for pos, size in enumerate(SYMBOL_SIZES):
        high, low = prefix[pos][end], prefix[pos][start]
        vectors.append([(high[k] - low[k] + strength / size) / (window + strength) for k in range(size)])
    return vectors


def _score_range(labels, prefix, model, start, end):
    losses, hits, brier_scores = [], [], []
    for j in range(start, end):
        vectors = _vectors_at(prefix, model, j)
        actual = labels[j]
        losses.append(sum(-math.log(max(v[a], 1e-15)) for v, a in zip(vectors, actual)) / 7)
        hits.append(sum(int(max(range(len(v)), key=lambda k: v[k]) == a)
                        for v, a in zip(vectors, actual)))
        brier_scores.append(sum(
            sum((p - int(k == a)) ** 2 for k, p in enumerate(v))
            for v, a in zip(vectors, actual)
        ) / 7)
    log_loss_ci_low, log_loss_ci_high = _mean_95_interval(losses)
    return {
        'mean_log_loss': sum(losses) / len(losses) if losses else None,
        'log_loss_ci_low': log_loss_ci_low,
        'log_loss_ci_high': log_loss_ci_high,
        'mean_symbol_hits': sum(hits) / len(hits) if hits else None,
        'mean_brier_score': sum(brier_scores) / len(brier_scores) if brier_scores else None,
        'cases': len(losses),
    }


def walkforward(history, min_train=20, max_cases=200, z=SIGNIFICANCE_Z):
    """Select on an earlier validation window and report a later holdout.

    Returns ``(chosen_model, scores)``. ``chosen_model`` is ``'uniform'`` unless a
    candidate significantly beats uniform on the validation window.
    """
    evaluable = min(max_cases, max(0, len(history) - min_train))
    validation_cases = evaluable // 2
    validation_start = len(history) - evaluable
    test_start = validation_start + validation_cases
    labels = _labels(history)
    prefix = _prefix_counts(labels)
    scored = {}
    for name in MODELS:
        validation = _score_range(labels, prefix, name, validation_start, test_start)
        test = _score_range(labels, prefix, name, test_start, len(history))
        scored[name] = {
            'validation_mean_log_loss': validation['mean_log_loss'],
            'validation_log_loss_95ci_low': validation['log_loss_ci_low'],
            'validation_log_loss_95ci_high': validation['log_loss_ci_high'],
            'validation_beats_uniform': name != 'uniform' and significantly_better(
                validation['mean_log_loss'], validation['log_loss_ci_low'],
                validation['log_loss_ci_high'], z),
            'validation_mean_symbol_hits': validation['mean_symbol_hits'],
            'validation_mean_brier_score': validation['mean_brier_score'],
            'validation_cases': validation['cases'],
            'test_mean_log_loss': test['mean_log_loss'],
            'test_log_loss_95ci_low': test['log_loss_ci_low'],
            'test_log_loss_95ci_high': test['log_loss_ci_high'],
            'test_mean_symbol_hits': test['mean_symbol_hits'],
            'test_mean_brier_score': test['mean_brier_score'],
            'test_cases': test['cases'],
        }
    eligible = (n for n in MODELS
                if n != 'uniform' and scored[n]['validation_cases'] >= 20
                and scored[n]['validation_beats_uniform'])
    chosen = min(eligible, key=lambda n: scored[n]['validation_mean_log_loss'], default='uniform')
    return chosen, scored