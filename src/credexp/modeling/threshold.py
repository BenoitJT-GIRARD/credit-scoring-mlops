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


def split_threshold_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    random_state: int = 0,
) -> dict:
    """Quantify how optimistic it is to pick a threshold on the data that scores it.

    Splits the sample into two stratified halves. `optimistic` picks and evaluates the
    threshold on the same half — what run_cv does today. `honest` picks on one half and
    evaluates on the other. The gap between them is the optimism, in the same units as
    the business cost, per evaluated row.
    """
    from sklearn.model_selection import train_test_split

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    idx_a, idx_b = train_test_split(
        np.arange(len(y_true)), test_size=0.5, random_state=random_state, stratify=y_true
    )

    thr_a, cost_a = find_best_threshold(y_true[idx_a], y_proba[idx_a], cost_fn, cost_fp)
    thr_b, _ = find_best_threshold(y_true[idx_b], y_proba[idx_b], cost_fn, cost_fp)

    optimistic = cost_a / len(idx_a)
    honest = business_cost(y_true[idx_a], y_proba[idx_a], thr_b, cost_fn, cost_fp) / len(idx_a)

    return {
        "optimistic": float(optimistic),
        "honest": float(honest),
        "optimism": float(honest - optimistic),
        "threshold_a": float(thr_a),
        "threshold_b": float(thr_b),
    }
