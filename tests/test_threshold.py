import numpy as np

from credexp.modeling.threshold import find_best_threshold


def test_find_best_threshold_returns_valid_range():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    thr, cost = find_best_threshold(y_true, y_proba)
    assert 0.0 < thr < 1.0
    assert cost >= 0.0
