"""Walk-forward statistical baselines; no claim of lottery predictability."""
import math
from collections import Counter
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
    vecs = [observed_probs(letters, model, 26)]
    for pos in range(6):
        vecs.append(observed_probs([int(r['digits'][pos]) for r in history], model, 10))
    return vecs


def walkforward(history, min_train=20, max_cases=200):
    """Each test draw is predicted strictly using draws preceding it."""
    scored = {}
    for name in MODELS:
        losses = []; hits = []
        for j in range(max(min_train, len(history) - max_cases), len(history)):
            vectors = probability_vectors(history[:j], name)
            actual = [ALPHABET.index(history[j]['letter'])] + [int(x) for x in history[j]['digits']]
            losses.append(sum(-math.log(max(v[a], 1e-15)) for v,a in zip(vectors, actual))/7)
            hits.append(sum(int(max(range(len(v)), key=lambda k:v[k]) == a) for v,a in zip(vectors,actual)))
        scored[name] = {'mean_log_loss': sum(losses)/len(losses) if losses else None,
                        'mean_symbol_hits': sum(hits)/len(hits) if hits else None,
                        'cases':len(losses)}
    chosen = min((n for n in MODELS if scored[n]['cases']), key=lambda n:scored[n]['mean_log_loss'], default='uniform')
    # An apparent difference on tiny samples is not trustworthy: conservative fallback.
    if scored[chosen]['cases'] < 20: chosen = 'uniform'
    return chosen, scored


