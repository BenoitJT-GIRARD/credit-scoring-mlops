"""The probability scale, and how much the threshold moves when the split does."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from credexp.modeling.calibration import (
    brier,
    calibrate_isotonic,
    expected_calibration_error,
    reliability,
)
from credexp.modeling.threshold import threshold_spread


def _calibrated(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that mean what they say: outcomes drawn at the stated rate."""
    rng = np.random.default_rng(seed)
    # A narrow band, so that inflating it four-fold stays below 1.0. Clipping at 1.0 would
    # create ties and change the ranking, which is the one thing these tests must isolate.
    p = rng.uniform(0.02, 0.20, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_a_score_that_means_what_it_says_has_near_zero_error() -> None:
    y, p = _calibrated()
    assert expected_calibration_error(y, p) < 0.02


def test_a_score_inflated_by_class_weighting_is_caught() -> None:
    """The failure mode of this project's model: ranking intact, scale multiplied.

    The real one overstates by a factor of 4.6 — a mean score of 0.366 against a base rate
    of 0.079 — and ROC AUC does not move by a thousandth, because multiplying every score
    by a constant reorders nothing.
    """
    y, p = _calibrated()
    inflated = p * 4.0

    assert roc_auc_score(y, inflated) == pytest.approx(roc_auc_score(y, p))
    assert expected_calibration_error(y, inflated) > 10 * expected_calibration_error(y, p)


def test_isotonic_recalibration_repairs_the_scale(caplog) -> None:
    y, p = _calibrated()
    inflated = p * 4.0
    fit, test = slice(None, 10_000), slice(10_000, None)

    recalibrate = calibrate_isotonic(y[fit], inflated[fit])
    repaired = recalibrate(inflated[test])

    assert expected_calibration_error(y[test], repaired) < expected_calibration_error(
        y[test], inflated[test]
    )
    assert brier(y[test], repaired) < brier(y[test], inflated[test])


def test_recalibration_never_inverts_the_ranking() -> None:
    """Isotonic is non-decreasing, so it can merge two scores but never swap them.

    Merging is not nothing: a step function turns distinguishable pairs into ties, and AUC
    counts a tie as half a point. So the honest claim is not that AUC is unchanged — it is
    that no pair is reordered, which is what makes the recalibrated score usable with a
    threshold mapped through the same function.
    """
    y, p = _calibrated(n=4000, seed=2)
    inflated = p * 3.0

    recalibrate = calibrate_isotonic(y, inflated)
    repaired = recalibrate(inflated)

    order = np.argsort(inflated)
    assert np.all(np.diff(repaired[order]) >= -1e-12), "a pair was reordered"
    assert roc_auc_score(y, repaired) == pytest.approx(roc_auc_score(y, inflated), abs=0.01)


def test_the_calibrator_only_sees_the_data_it_was_given() -> None:
    """Fitting it on the holdout would make the holdout number meaningless."""
    rng = np.random.default_rng(3)
    y_fit = rng.integers(0, 2, 500)
    p_fit = rng.uniform(size=500)

    recalibrate = calibrate_isotonic(y_fit, p_fit)
    out = recalibrate(np.array([-5.0, 0.5, 5.0]))

    assert ((out >= 0.0) & (out <= 1.0)).all(), "out-of-range inputs are clipped, not extrapolated"


def test_the_bins_hold_equal_population() -> None:
    frame = reliability(np.array([0, 1] * 50), np.linspace(0, 1, 100), n_bins=10)
    assert len(frame) == 10
    assert frame["count"].nunique() == 1


def test_the_gap_is_signed_so_over_and_under_statement_differ() -> None:
    y = np.array([0] * 92 + [1] * 8)
    always_high = np.full(100, 0.4)

    frame = reliability(y, always_high, n_bins=2)
    assert (frame["gap"] < 0).all(), "claiming 0.4 on an 8% base rate overstates the risk"


def test_an_empty_input_is_nan_rather_than_an_exception() -> None:
    empty = np.array([], dtype=float)
    assert np.isnan(brier(empty, empty))
    assert np.isnan(expected_calibration_error(empty, empty))
    assert reliability(empty, empty).empty


def test_the_threshold_spread_is_an_interval_around_its_median() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(size=2000) * 0.6, 0, 1)

    out = threshold_spread(y, p, n_repeats=15, random_state=1)

    assert out["n_repeats"] == 15
    assert out["ci_low"] <= out["median"] <= out["ci_high"]


def test_a_perfectly_separable_problem_gives_a_tight_spread() -> None:
    """A stable threshold and an unstable one must not report the same interval."""
    y = np.array([0] * 500 + [1] * 500)
    clean = np.array([0.1] * 500 + [0.9] * 500)
    noisy = np.concatenate(
        [np.random.default_rng(0).uniform(size=500), np.random.default_rng(1).uniform(size=500)]
    )

    tight = threshold_spread(y, clean, n_repeats=15, random_state=0)
    loose = threshold_spread(y, noisy, n_repeats=15, random_state=0)

    assert (tight["ci_high"] - tight["ci_low"]) < (loose["ci_high"] - loose["ci_low"])


def test_a_single_class_sample_is_skipped_rather_than_scored() -> None:
    out = threshold_spread(np.zeros(50, dtype=int), np.random.default_rng(0).uniform(size=50))
    assert out["n_repeats"] == 0
    assert np.isnan(out["median"])
