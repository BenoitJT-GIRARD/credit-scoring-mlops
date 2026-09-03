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


def threshold_spread(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    n_repeats: int = 25,
    fraction: float = 0.5,
    random_state: int = 0,
) -> dict:
    """How much the chosen threshold moves when the split it was chosen on changes.

    The shipped threshold is picked on one validation split. That makes it a point
    estimate, and a point estimate presented as a decision hides the question a reader
    should ask: are 0.42 and 0.48 two different settings, or the same number twice?

    ``n_repeats`` resamples of half the data, the threshold reselected on each. The spread
    of those is the answer. It is deliberately not a bootstrap of the cost — the quantity
    that has to be stable is the threshold itself, because that is what gets deployed.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(random_state)
    size = max(int(len(y_true) * fraction), 2)

    thresholds = []
    for _ in range(n_repeats):
        index = rng.choice(len(y_true), size=size, replace=False)
        if len(set(y_true[index].tolist())) < 2:
            continue
        threshold, _ = find_best_threshold(y_true[index], y_proba[index], cost_fn, cost_fp)
        thresholds.append(threshold)

    if not thresholds:
        return {
            "median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_repeats": 0,
        }

    drawn = np.asarray(thresholds)
    return {
        "median": float(np.median(drawn)),
        "ci_low": float(np.percentile(drawn, 2.5)),
        "ci_high": float(np.percentile(drawn, 97.5)),
        "n_repeats": len(drawn),
    }


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


def threshold_shift(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    shipped_threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """Compare the shipped threshold to the one this sample would have chosen.

    The optimal threshold is computed for information only. Adopting it would spend the
    holdout, which exists precisely so that no decision is fitted to it.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    optimal_threshold, optimal_cost = find_best_threshold(y_true, y_proba, cost_fn, cost_fp)
    shipped_cost = business_cost(y_true, y_proba, shipped_threshold, cost_fn, cost_fp)

    return {
        "shipped_threshold": float(shipped_threshold),
        "optimal_threshold": float(optimal_threshold),
        "shift": float(optimal_threshold - shipped_threshold),
        "shipped_cost": float(shipped_cost / n),
        "optimal_cost": float(optimal_cost / n),
        "regret": float((shipped_cost - optimal_cost) / n),
    }
