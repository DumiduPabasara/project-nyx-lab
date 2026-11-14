"""Optional CPU sequence models for Nyx 2.0.

These are experimental baselines, not evidence that draws are predictable.
Training/selection uses chronological splits only. The final test window is never
used to select a model.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

from model import ALPHABET
from prizes import evaluate_prize

try:  # Keep the existing collector/dashboard usable without PyTorch.
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - depends on deployment environment
    np = None
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

LOOKBACK = 12
MIN_HISTORY = 380  # aligns the neural validation/test windows with model.walkforward()
MAX_CASES = 200
HIDDEN_SIZE = 24
EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 32
SYMBOL_SIZES = (len(ALPHABET), 10, 10, 10, 10, 10, 10)
FEATURE_SIZE = sum(SYMBOL_SIZES)


def available() -> bool:
    return torch is not None and np is not None


def _reason_unavailable(history_length: int) -> str | None:
    if not available():
        return "PyTorch/NumPy is not installed in the runner; classical models remain active."
    if history_length < MIN_HISTORY:
        return f"Need at least {MIN_HISTORY} historical draws for the experimental neural holdout; currently have {history_length}."
    return None


def _row_labels(row: dict[str, Any]) -> list[int]:
    letter = str(row["letter"]).upper()
    digits = str(row["digits"])
    if letter not in ALPHABET or len(digits) != 6 or not digits.isdigit():
        raise ValueError("Invalid draw supplied to Nyx sequence model")
    return [ALPHABET.index(letter)] + [int(char) for char in digits]


def encode_rows(rows: list[dict[str, Any]]):
    """One-hot encode each draw into 85 features (25 letter + 60 digit)."""
    if not available():
        raise RuntimeError("PyTorch and NumPy are required for sequence models")
    encoded = np.zeros((len(rows), FEATURE_SIZE), dtype=np.float32)
    offsets = np.cumsum((0,) + SYMBOL_SIZES[:-1]).tolist()
    for row_index, row in enumerate(rows):
        for position, symbol in enumerate(_row_labels(row)):
            encoded[row_index, offsets[position] + symbol] = 1.0
    return encoded


def _make_training_arrays(rows: list[dict[str, Any]]):
    features = encode_rows(rows)
    labels = np.asarray([_row_labels(row) for row in rows], dtype=np.int64)
    if len(rows) <= LOOKBACK:
        raise ValueError("Not enough rows to create sequence examples")
    x = np.stack([features[i - LOOKBACK:i] for i in range(LOOKBACK, len(rows))])
    y = labels[LOOKBACK:]
    return x, y


if available():
    class _SequenceNetwork(nn.Module):
        def __init__(self, architecture: str):
            super().__init__()
            if architecture == "lstm":
                self.recurrent = nn.LSTM(FEATURE_SIZE, HIDDEN_SIZE, batch_first=True)
            elif architecture == "gru":
                self.recurrent = nn.GRU(FEATURE_SIZE, HIDDEN_SIZE, batch_first=True)
            else:
                raise ValueError("Unsupported sequence architecture")
            self.norm = nn.LayerNorm(HIDDEN_SIZE)
            self.dropout = nn.Dropout(0.20)
            self.heads = nn.ModuleList(nn.Linear(HIDDEN_SIZE, size) for size in SYMBOL_SIZES)

        def forward(self, x):
            output, _ = self.recurrent(x)
            state = self.dropout(self.norm(output[:, -1, :]))
            return [head(state) for head in self.heads]
else:  # pragma: no cover - exercised only when optional dependency is absent
    _SequenceNetwork = None


def _seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)


def fit_sequence_model(rows: list[dict[str, Any]], architecture: str):
    """Fit on an initial chronological segment with a trailing validation slice."""
    if not available():
        raise RuntimeError("PyTorch and NumPy are required for sequence models")
    if len(rows) < 180:
        raise ValueError("At least 180 training draws are required")
    _seed_everything()
    x, y = _make_training_arrays(rows)
    split = max(1, int(len(x) * 0.85))
    if split < 80 or len(x) - split < 10:
        raise ValueError("Not enough training examples for a stable chronological split")

    train_x = torch.from_numpy(x[:split])
    train_y = torch.from_numpy(y[:split])
    valid_x = torch.from_numpy(x[split:])
    valid_y = torch.from_numpy(y[split:])
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)

    network = _SequenceNetwork(architecture)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=0.01)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(EPOCHS):
        network.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = network(batch_x)
            loss = sum(F.cross_entropy(logits[i], batch_y[:, i]) for i in range(7)) / 7
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

        network.eval()
        with torch.no_grad():
            valid_logits = network(valid_x)
            valid_loss = sum(
                F.cross_entropy(valid_logits[i], valid_y[:, i]).item()
                for i in range(7)
            ) / 7
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = copy.deepcopy(network.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("Neural model training did not produce a valid checkpoint")
    network.load_state_dict(best_state)
    network.eval()
    return network, {"best_validation_loss": float(best_loss), "best_epoch": best_epoch}


def predict_probabilities(network, context: list[dict[str, Any]]) -> list[list[float]]:
    if len(context) < LOOKBACK:
        raise ValueError(f"At least {LOOKBACK} prior draws are needed for a sequence forecast")
    features = encode_rows(context[-LOOKBACK:])
    x = torch.from_numpy(features).unsqueeze(0)
    network.eval()
    with torch.no_grad():
        logits = network(x)
        return [torch.softmax(item[0], dim=-1).cpu().numpy().astype(float).tolist() for item in logits]


def _mean_ci(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return mean - margin, mean + margin


def _score_predictions(predictions: list[list[list[float]]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses, briers, hits, prize_hits, reference_prizes = [], [], [], [], []
    for vectors, actual_row in zip(predictions, rows):
        actual = _row_labels(actual_row)
        losses.append(sum(-math.log(max(v[a], 1e-15)) for v, a in zip(vectors, actual)) / 7)
        briers.append(sum(
            sum((p - int(k == a)) ** 2 for k, p in enumerate(v))
            for v, a in zip(vectors, actual)
        ) / 7)
        chosen = [max(range(len(v)), key=lambda k: v[k]) for v in vectors]
        hits.append(sum(int(p == a) for p, a in zip(chosen, actual)))
        prize = evaluate_prize(ALPHABET[chosen[0]], "".join(map(str, chosen[1:])),
                               actual_row["letter"], actual_row["digits"])
        prize_hits.append(int(prize.amount > 0))
        reference_prizes.append(prize.amount)
    low, high = _mean_ci(losses)
    return {
        "mean_log_loss": sum(losses) / len(losses) if losses else None,
        "log_loss_ci_low": low,
        "log_loss_ci_high": high,
        "mean_brier_score": sum(briers) / len(briers) if briers else None,
        "mean_symbol_hits": sum(hits) / len(hits) if hits else None,
        "prize_hit_count": sum(prize_hits),
        "mean_reference_prize_rs": sum(reference_prizes) / len(reference_prizes) if reference_prizes else None,
        "cases": len(losses),
    }


def _evaluate(network, history, start: int, end: int) -> dict[str, Any]:
    forecasts = [predict_probabilities(network, history[:index]) for index in range(start, end)]
    return _score_predictions(forecasts, history[start:end])


def evaluate_deep_models(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare LSTM, GRU and their probability ensemble on validation + holdout."""
    reason = _reason_unavailable(len(history))
    if reason:
        return {"available": False, "reason": reason, "minimum_history": MIN_HISTORY,
                "history_draws": len(history), "scores": {}}

    evaluable = min(MAX_CASES, max(0, len(history) - 180))
    validation_cases = evaluable // 2
    validation_start = len(history) - evaluable
    test_start = validation_start + validation_cases
    if validation_start < 180 or validation_cases < 20 or len(history) - test_start < 20:
        return {"available": False, "reason": "Not enough independent validation and test cases.",
                "minimum_history": MIN_HISTORY, "history_draws": len(history), "scores": {}}

    architectures = ("lstm", "gru")
    validation_networks = {}
    scores = {}
    try:
        for name in architectures:
            network, training = fit_sequence_model(history[:validation_start], name)
            validation_networks[name] = network
            scores[name] = {
                "validation": _evaluate(network, history, validation_start, test_start),
                "training": training,
            }

        validation_ensemble = []
        for index in range(validation_start, test_start):
            left = predict_probabilities(validation_networks["lstm"], history[:index])
            right = predict_probabilities(validation_networks["gru"], history[:index])
            validation_ensemble.append([[(a + b) / 2 for a, b in zip(x, y)] for x, y in zip(left, right)])
        scores["ensemble"] = {"validation": _score_predictions(validation_ensemble, history[validation_start:test_start])}

        # The final test window is evaluated only after the validation-based choice is fixed.
        test_networks = {}
        for name in architectures:
            network, training = fit_sequence_model(history[:test_start], name)
            test_networks[name] = network
            scores[name]["test"] = _evaluate(network, history, test_start, len(history))
            scores[name]["test_training"] = training

        test_ensemble = []
        for index in range(test_start, len(history)):
            left = predict_probabilities(test_networks["lstm"], history[:index])
            right = predict_probabilities(test_networks["gru"], history[:index])
            test_ensemble.append([[(a + b) / 2 for a, b in zip(x, y)] for x, y in zip(left, right)])
        scores["ensemble"]["test"] = _score_predictions(test_ensemble, history[test_start:])

        selected = min(scores, key=lambda name: scores[name]["validation"]["mean_log_loss"])
        return {
            "available": True,
            "reason": None,
            "selected_model": selected,
            "lookback_draws": LOOKBACK,
            "history_draws": len(history),
            "validation_start_draw": history[validation_start]["draw_number"],
            "test_start_draw": history[test_start]["draw_number"],
            "validation_cases": test_start - validation_start,
            "test_cases": len(history) - test_start,
            "selection_metric": "validation mean log loss (lower is better)",
            "scores": scores,
        }
    except Exception as exc:
        # A neural-model failure must never stop the established statistical pipeline.
        return {"available": False, "reason": f"Neural evaluation failed ({type(exc).__name__}); classical models remain active.",
                "minimum_history": MIN_HISTORY, "history_draws": len(history), "scores": {}}


def predict_next(history: list[dict[str, Any]], model_name: str) -> list[list[float]]:
    """Fit the selected sequence architecture on all known draws and forecast one step."""
    if model_name == "ensemble":
        lstm, _ = fit_sequence_model(history, "lstm")
        gru, _ = fit_sequence_model(history, "gru")
        left = predict_probabilities(lstm, history)
        right = predict_probabilities(gru, history)
        return [[(a + b) / 2 for a, b in zip(x, y)] for x, y in zip(left, right)]
    if model_name not in ("lstm", "gru"):
        raise ValueError("Unknown Nyx neural model")
    network, _ = fit_sequence_model(history, model_name)
    return predict_probabilities(network, history)
