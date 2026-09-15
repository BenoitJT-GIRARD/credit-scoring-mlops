"""The numeric branch, and the order its steps have to be in.

Three of the four steps only work in one order: infinities become missing before the
imputer runs, or the imputer passes them on; clipping happens after scaling, or it clips
raw incomes instead of standard deviations. The assembly is three lines long and the order
inside it is the whole of what it decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credexp.modeling.pipelines import make_numeric_steps


def test_the_unscaled_branch_is_the_two_steps_a_tree_needs() -> None:
    """A gradient-boosted tree is invariant to a monotone rescaling, so it gets neither."""
    names = [name for name, _ in make_numeric_steps(scale=False)]

    assert names == ["inf_to_nan", "imputer"]


def test_the_scaled_branch_clips_after_it_scales() -> None:
    """Clipping to ±10 means ten standard deviations, which is only true after the scaler."""
    names = [name for name, _ in make_numeric_steps(scale=True)]

    assert names == ["inf_to_nan", "imputer", "scaler", "clip"]


def test_the_infinity_step_comes_before_the_imputer() -> None:
    """The imputer does not treat an infinity as missing: it hands it to the estimator."""
    steps = make_numeric_steps(scale=False)
    frame = pd.DataFrame({"ratio": [1.0, np.inf, 3.0]})

    out = Pipeline(steps).fit_transform(frame)

    assert np.isfinite(out).all()
    # The median of the two finite values, which is what the imputer put in place of the
    # infinity once the step before it had called that value missing.
    assert out.ravel().tolist() == [1.0, 2.0, 3.0]


def test_the_median_is_the_imputation_and_not_the_mean() -> None:
    """Income on this dataset has a tail that drags a mean and leaves a median alone."""
    steps = make_numeric_steps(scale=False)
    frame = pd.DataFrame({"income": [1.0, 2.0, 3.0, 1_000_000.0, np.nan]})

    out = Pipeline(steps).fit_transform(frame).ravel().tolist()

    assert out[-1] == 2.5
