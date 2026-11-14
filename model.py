"""Walk-forward statistical baselines; no claim of lottery predictability."""
import math
from collections import Counter
ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MODELS = ("uniform", "all_history", "recent_30", "recent_100")

def observed_probs(rows, model, alphabet_size):
    if model == 'uniform' or not rows:
        return [1 / alphabet_size] * alphabet_size
    if model == 'recent_30': rows = rows[-30:]
    elif model == 'recent_100': rows = rows[-100:]
    # Dirichlet smoothing means limited data don't produce false certainty.
    counts = Counter(rows)
    strength = 20.0
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


def _score_range(history, model, start, end):
    losses = []
    hits = []
    brier_scores = []
    for j in range(start, end):
        vectors = probability_vectors(history[:j], model)
        actual = [ALPHABET.index(history[j]['letter'])] + [int(x) for x in history[j]['digits']]
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


def walkforward(history, min_train=20, max_cases=200):
    """Select on an earlier validation window and report a later holdout."""
    evaluable = min(max_cases, max(0, len(history) - min_train))
    validation_cases = evaluable // 2
    validation_start = len(history) - evaluable
    test_start = validation_start + validation_cases
    scored = {}
    for name in MODELS:
        validation = _score_range(history, name, validation_start, test_start)
        test = _score_range(history, name, test_start, len(history))
        scored[name] = {
            'validation_mean_log_loss': validation['mean_log_loss'],
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
    eligible = (n for n in MODELS if scored[n]['validation_cases'] >= 20)
    chosen = min(eligible, key=lambda n:scored[n]['validation_mean_log_loss'], default='uniform')
    return chosen, scored
