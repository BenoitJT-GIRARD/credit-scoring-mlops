"""Two transformers, and the reason each of them is a pipeline step rather than a cleanup.

Both exist because of something the Home Credit features do. A ratio whose denominator is
zero returns an infinity, and `SimpleImputer` does not consider an infinity missing: it
passes it straight through to the estimator. And one applicant with an income three orders
of magnitude above the rest moves a scaler more than the three hundred thousand others
together.

Fitted inside the pipeline, both are applied to the validation fold with what the training
fold taught them, which is the point of making them steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credexp.modeling.preprocess import Clipper, InfToNan


def test_infinities_become_missing_values_in_a_frame() -> None:
    frame = pd.DataFrame({"ratio": [1.0, np.inf, -np.inf, 4.0]})

    out = InfToNan().fit_transform(frame)

    assert isinstance(out, pd.DataFrame)
    assert out["ratio"].isna().sum() == 2
    assert out["ratio"].to_list()[0] == 1.0


def test_infinities_become_missing_values_in_an_array() -> None:
    """The array path exists because a pipeline step before this one may hand over one."""
    out = InfToNan().fit_transform(np.array([[1.0, np.inf], [-np.inf, 4.0]]))

    assert isinstance(out, np.ndarray)
    assert np.isnan(out).sum() == 2


def test_the_transformer_keeps_the_column_names_it_was_given() -> None:
    """A step that turns a frame into an array is how a model loses its feature names."""
    frame = pd.DataFrame({"a": [1.0], "b": [np.inf]})

    out = InfToNan().fit_transform(frame)

    assert list(out.columns) == ["a", "b"]


def test_clipping_holds_both_ends_of_the_range() -> None:
    frame = pd.DataFrame({"z": [-50.0, -1.0, 0.0, 1.0, 50.0]})

    out = Clipper(-10.0, 10.0).fit_transform(frame)

    assert out["z"].min() == -10.0
    assert out["z"].max() == 10.0
    assert out["z"].to_list()[1:4] == [-1.0, 0.0, 1.0]


def test_clipping_an_array_gives_back_an_array() -> None:
    out = Clipper(-2.0, 2.0).fit_transform(np.array([[-9.0, 0.5, 9.0]]))

    assert isinstance(out, np.ndarray)
    assert out.tolist() == [[-2.0, 0.5, 2.0]]


def test_both_transformers_fit_on_nothing_and_so_leak_nothing() -> None:
    """Neither keeps a statistic of the data it saw, which is what makes them safe anywhere.

    A scaler between them would not be: it learns a mean and a standard deviation, and
    fitting it outside the fold is the most common leak on this dataset.
    """
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    for transformer in (InfToNan(), Clipper()):
        fitted = transformer.fit(frame)
        assert fitted is transformer
        assert not [name for name in vars(fitted) if name.endswith("_")]
