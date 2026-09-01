"""How much the published cost depends on assumptions and on this particular sample.

The business cost rests on a ratio nobody measured and on one holdout drawn once. Both
deserve a number rather than a footnote.
"""

from __future__ import annotations

import numpy as np

from credexp.modeling.threshold import business_cost, find_best_threshold


def cost_ratio_sweep(y_true, y_proba, ratios=(1, 2, 3, 5, 10, 15, 20, 30, 50)) -> list[dict]:
    """Optimal threshold and cost per row for each false-negative to false-positive ratio."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    rows = []
    for ratio in ratios:
        threshold, cost = find_best_threshold(y_true, y_proba, cost_fn=float(ratio), cost_fp=1.0)
        rows.append(
            {"ratio": ratio, "threshold": float(threshold), "cost_per_row": float(cost / n)}
        )
    return rows


def bootstrap_cost_ci(
    y_true,
    y_proba,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
    n_resamples: int = 1000,
    random_state: int = 0,
) -> dict:
    """Percentile interval for the cost per row, at a fixed threshold.

    The threshold is held fixed on purpose: resampling and re-optimising at once would
    mix two sources of variation and produce an interval nobody could interpret.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    costs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        costs[i] = business_cost(y_true[idx], y_proba[idx], threshold, cost_fn, cost_fp) / n

    return {
        "median": float(np.median(costs)),
        "low": float(np.percentile(costs, 2.5)),
        "high": float(np.percentile(costs, 97.5)),
        "n_resamples": int(n_resamples),
    }
