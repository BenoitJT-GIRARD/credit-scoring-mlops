"""Whether a probability from this model means what it says.

The API returns a probability and the business threshold is a point on that scale. If the
scale is wrong, the threshold does not transport: a cut at 0.42 fitted on one split will
accept a different population on the next, for reasons that have nothing to do with the
applicants.

ROC AUC cannot see this. A model whose ranking is perfect and whose scale is squashed
scores 1.0 and still tells a credit officer the wrong number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

__all__ = ["brier", "calibrate_isotonic", "expected_calibration_error", "reliability"]


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean squared error between the stated probability and the outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_score - y_true) ** 2))


def reliability(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed default rate against predicted probability, in equal-population bins.

    Equal population rather than equal width. On a problem with an 8% base rate the scores
    pile up near zero, and equal-width bins leave the upper half nearly empty — the
    calibration error then averages over noise where it should be measuring.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin", "count", "mean_score", "observed", "gap"])

    order = np.argsort(y_score, kind="stable")
    rows = []
    for index, chunk in enumerate(np.array_split(order, min(n_bins, len(order)))):
        if len(chunk) == 0:
            continue
        predicted = float(y_score[chunk].mean())
        observed = float(y_true[chunk].mean())
        rows.append(
            {
                "bin": index,
                "count": len(chunk),
                "mean_score": predicted,
                "observed": observed,
                "gap": observed - predicted,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Average distance between the stated probability and the observed frequency."""
    frame = reliability(y_true, y_score, n_bins)
    if frame.empty:
        return float("nan")
    weights = frame["count"].to_numpy(dtype=float)
    return float(np.average(np.abs(frame["gap"].to_numpy()), weights=weights))


def calibrate_isotonic(y_val: np.ndarray, p_val: np.ndarray):
    """Fit a monotone recalibration on validation scores, and return it as a callable.

    Isotonic rather than Platt: it makes no assumption about the shape of the distortion,
    which matters because a gradient-boosted model's miscalibration is rarely a sigmoid.
    Being monotone, it cannot change the ranking — so ROC AUC is untouched by construction
    and only the scale moves. That is the whole point: if a recalibrated model scored
    differently on AUC, something other than calibration would have happened.

    It is fitted on the validation split alone. Fitting it on the holdout would make the
    holdout number meaningless, which is the mistake this function exists to avoid.
    """
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(p_val, dtype=float), np.asarray(y_val, dtype=float))

    def apply(scores: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(np.asarray(scores, dtype=float)), dtype=float)

    return apply
