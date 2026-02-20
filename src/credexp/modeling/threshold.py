from __future__ import annotations

import numpy as np


def business_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return float(fn * cost_fn + fp * cost_fp)


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [business_cost(y_true, y_proba, t, cost_fn, cost_fp) for t in thresholds]
    best_i = int(np.argmin(costs))
    return float(thresholds[best_i]), float(costs[best_i])
