"""Per-group behaviour of the scoring decision.

A credit model decides access to funding. Publishing an aggregate cost without ever
looking at how it falls across groups leaves the most consequential question unasked.
Nothing here corrects anything: it measures, so the gap can be stated.
"""

from __future__ import annotations

import numpy as np

from credexp.modeling.threshold import business_cost

BAND_EDGES = (30, 40, 50, 60)
BAND_LABELS = ("<30", "30-39", "40-49", "50-59", "60+")


def age_bands(days_birth) -> np.ndarray:
    """Turn Home Credit's DAYS_BIRTH (negative, counted backwards) into decade labels."""
    years = np.abs(np.asarray(days_birth, dtype=float)) / 365.25
    return np.asarray(BAND_LABELS)[np.digitize(years, BAND_EDGES)]


def _rate(numerator: int, denominator: int) -> float | None:
    """None rather than 0.0 on an empty denominator: unknown is not perfect."""
    return float(numerator / denominator) if denominator else None


def group_report(
    y_true,
    y_proba,
    groups,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> list[dict]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    groups = np.asarray(groups)

    rows = []
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        y_g, p_g = y_true[mask], y_proba[mask]
        predicted = p_g >= threshold
        n = int(mask.sum())

        positives = int(y_g.sum())
        negatives = n - positives
        false_negatives = int(((~predicted) & (y_g == 1)).sum())
        false_positives = int((predicted & (y_g == 0)).sum())

        rows.append(
            {
                "group": str(name),
                "n": n,
                "default_rate": _rate(positives, n),
                "refusal_rate": _rate(int(predicted.sum()), n),
                "fnr": _rate(false_negatives, positives),
                "fpr": _rate(false_positives, negatives),
                "cost_per_row": (
                    float(business_cost(y_g, p_g, threshold, cost_fn, cost_fp) / n) if n else None
                ),
            }
        )
    return rows
