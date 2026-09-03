"""Choosing the threshold, and pricing the two shortcuts that were taken while choosing it.

`split_threshold_cost` measures selecting the threshold on the fold that then scores it;
`threshold_shift` measures calibrating on one model and applying to another. Both have a
case where the answer must be zero -- a bias estimator that never returns zero is measuring
its own noise.
"""

import numpy as np

from credexp.modeling.threshold import (
    find_best_threshold,
    split_threshold_cost,
    threshold_shift,
)


def test_find_best_threshold_returns_valid_range():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    thr, cost = find_best_threshold(y_true, y_proba)
    assert 0.0 < thr < 1.0
    assert cost >= 0.0


def test_split_threshold_cost_reports_both_estimates_and_their_gap() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, size=400)
    # Probabilities correlated with the label but noisy, so a threshold tuned on one
    # half is genuinely a little wrong on the other.
    proba = np.clip(y * 0.4 + rng.normal(0.3, 0.2, size=400), 0.01, 0.99)

    out = split_threshold_cost(y, proba, cost_fn=10.0, cost_fp=1.0, random_state=0)

    assert out["honest"] >= out["optimistic"]
    assert out["optimism"] == out["honest"] - out["optimistic"]
    assert 0.0 <= out["threshold_a"] <= 1.0


def test_split_threshold_cost_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, size=300)
    proba = np.clip(y * 0.4 + rng.normal(0.3, 0.2, size=300), 0.01, 0.99)

    first = split_threshold_cost(y, proba, 10.0, 1.0, random_state=7)
    second = split_threshold_cost(y, proba, 10.0, 1.0, random_state=7)

    assert first == second


def test_threshold_shift_reports_the_regret_of_the_shipped_threshold() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.35, 0.65])

    out = threshold_shift(y, proba, shipped_threshold=0.49, cost_fn=10.0, cost_fp=1.0)

    assert out["shipped_threshold"] == 0.49
    assert out["regret"] >= 0.0
    assert out["shift"] == out["optimal_threshold"] - 0.49


def test_threshold_shift_has_no_regret_when_the_shipped_threshold_is_optimal() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])

    out = threshold_shift(y, proba, shipped_threshold=0.5, cost_fn=10.0, cost_fp=1.0)

    assert out["regret"] == 0.0
