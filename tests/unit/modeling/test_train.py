"""The cross-validation loop, and the one thing it refuses to let a fold do.

A fold that picks its own threshold and then scores itself with it reports a cost it
cannot achieve on anything else. The loop cycles them instead: fold k is scored with the
threshold fold k-1 chose. That is the property worth a test, because the bias it removes
is small enough to survive a reading — measured at 0.0029 per applicant on the holdout,
around 0.6 % of a cost near 0.49 — and it points the wrong way.

Everything here runs on a few hundred synthetic rows. What is under test is the protocol,
not the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from credexp.modeling.train import TrainConfig, _compute_sample_weight, _make_pipeline, run_cv


@pytest.fixture()
def separable() -> tuple[pd.DataFrame, pd.Series]:
    """Three hundred applicants, one informative column, eight per cent of them defaulting.

    The rate is the Home Credit rate, because a loop tested on a balanced sample never
    meets the case the class weights exist for.
    """
    rng = np.random.default_rng(7)
    n = 300
    target = (rng.random(n) < 0.08).astype(int)
    signal = target * 1.5 + rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({"signal": signal, "noise": rng.normal(0.0, 1.0, n)})
    return frame, pd.Series(target, name="TARGET")


# --- The class weights -----------------------------------------------------


def test_the_two_classes_end_up_weighing_the_same(separable) -> None:
    """`balanced` means the rare class carries as much total weight as the common one."""
    _, target = separable
    weights = _compute_sample_weight(target.to_numpy())

    positive = weights[target.to_numpy() == 1].sum()
    negative = weights[target.to_numpy() == 0].sum()
    assert positive == pytest.approx(negative)


def test_a_class_nobody_observed_does_not_divide_by_zero() -> None:
    """A stratified fold can be handed a slice with no default in it."""
    weights = _compute_sample_weight(np.zeros(10, dtype=int))

    assert np.isfinite(weights).all()
    assert weights.tolist() == [0.5] * 10


# --- The pipeline the loop fits --------------------------------------------


def test_a_tree_is_not_scaled_and_a_regression_is() -> None:
    """Scaling a gradient-boosted tree costs time and changes nothing it can see."""
    tree = [name for name, _ in _make_pipeline("lgbm", None, TrainConfig()).steps]
    regression = [name for name, _ in _make_pipeline("lr", None, TrainConfig()).steps]

    assert "scaler" not in tree
    assert "scaler" in regression


def test_undersampling_is_a_step_of_the_pipeline_and_not_a_pass_over_the_data() -> None:
    """Inside the pipeline it runs on the training fold only, which is the whole point."""
    pipeline = _make_pipeline("lr", None, TrainConfig(use_undersampling=True))

    assert isinstance(pipeline, ImbPipeline)
    assert [name for name, _ in pipeline.steps][-2:] == ["under", "model"]


def test_a_baseline_is_never_undersampled() -> None:
    """A dummy that predicts the majority class has no imbalance left to correct."""
    pipeline = _make_pipeline("dummy", "most_frequent", TrainConfig(use_undersampling=True))

    assert "under" not in [name for name, _ in pipeline.steps]


def test_an_unknown_model_name_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="xgboost"):
        _make_pipeline("xgboost", None, TrainConfig())


# --- What the loop reports --------------------------------------------------


def test_the_loop_reports_the_six_numbers_the_selection_reads(separable) -> None:
    features, target = separable

    report = run_cv(features, target, "lr", None, TrainConfig(n_splits=3))

    assert set(report) == {
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "best_threshold_mean",
        "business_cost_per_row",
    }
    assert 0.0 <= report["roc_auc_mean"] <= 1.0
    assert report["business_cost_per_row"] > 0.0


def test_the_loop_is_reproducible_at_a_fixed_seed(separable) -> None:
    """Two runs of the same configuration return the same six numbers, to the float."""
    features, target = separable
    config = TrainConfig(n_splits=3, random_state=11)

    first = run_cv(features, target, "lr", None, config)
    second = run_cv(features, target, "lr", None, config)

    assert first == second


def test_a_cost_ratio_of_ten_to_one_lowers_the_threshold(separable) -> None:
    """Ten refusals to one default: the loop should want to refuse more, not fewer."""
    features, target = separable

    symmetric = run_cv(features, target, "lr", None, TrainConfig(n_splits=3, cost_fn=1.0))
    asymmetric = run_cv(features, target, "lr", None, TrainConfig(n_splits=3, cost_fn=10.0))

    assert asymmetric["best_threshold_mean"] < symmetric["best_threshold_mean"]


def test_the_baseline_cannot_beat_the_model_on_cost(separable) -> None:
    """The comparison the README publishes, on three hundred rows instead of three hundred thousand."""
    features, target = separable
    config = TrainConfig(n_splits=3)

    model = run_cv(features, target, "lr", None, config)
    baseline = run_cv(features, target, "dummy", "most_frequent", config)

    assert model["business_cost_per_row"] < baseline["business_cost_per_row"]
    assert baseline["roc_auc_mean"] == pytest.approx(0.5, abs=0.05)
