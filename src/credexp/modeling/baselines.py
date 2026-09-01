"""What it costs to decide without a model.

On a sample with 8 % defaults and a false negative worth ten false positives, refusing
everyone is not an absurd policy. It is the floor the model has to beat to justify itself.
"""

from __future__ import annotations

import numpy as np


def trivial_baselines(
    y_true,
    cost_fn: float,
    cost_fp: float,
    random_state: int = 0,
    n_draws: int = 100,
) -> list[dict]:
    y_true = np.asarray(y_true)
    n = len(y_true)
    positives = int(y_true.sum())
    negatives = n - positives
    base_rate = positives / n if n else 0.0

    rng = np.random.default_rng(random_state)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        predicted = rng.random(n) < base_rate
        false_negatives = int(((~predicted) & (y_true == 1)).sum())
        false_positives = int((predicted & (y_true == 0)).sum())
        draws[i] = (false_negatives * cost_fn + false_positives * cost_fp) / n

    return [
        {
            "name": "accept-all",
            "cost_per_row": float(positives * cost_fn / n),
            "note": "Every defaulter is missed; no good applicant is refused.",
        },
        {
            "name": "reject-all",
            "cost_per_row": float(negatives * cost_fp / n),
            "note": "No defaulter is missed; every good applicant is refused.",
        },
        {
            "name": "random-at-base-rate",
            "cost_per_row": float(draws.mean()),
            "note": f"Mean of {n_draws} draws at the observed default rate.",
        },
    ]
