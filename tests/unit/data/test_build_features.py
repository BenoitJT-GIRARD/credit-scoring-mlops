"""The joins that turn seven Home Credit tables into one row per applicant.

None of the data is here, and none of it needs to be: what these pin is the handful of
decisions the build makes on the way, each of which is a way the feature matrix could be
silently wrong. A sentinel read as a number, a category encoded into a column nobody
named, a ratio with a zero denominator, a missing table that shortens the matrix by a
hundred columns without a word.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credexp.data.build_features import (
    REQUIRED_RAW_FILES,
    _check_raw_files,
    application_train_test,
    one_hot_encoder,
)

#: The columns `application_train_test` reads by name. A tiny frame carrying these is
#: enough to exercise every decision the function makes.
APPLICATION_COLUMNS = {
    "SK_ID_CURR": [1, 2, 3],
    "TARGET": [0, 1, 0],
    "CODE_GENDER": ["M", "F", "XNA"],
    "FLAG_OWN_CAR": ["Y", "N", "Y"],
    "FLAG_OWN_REALTY": ["Y", "Y", "N"],
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
    "DAYS_BIRTH": [-12000, -15000, -9000],
    # 365243 is Home Credit's « not employed » sentinel, and it is a positive number of
    # days in a column where every real value is negative.
    "DAYS_EMPLOYED": [-2000, 365243, -500],
    "AMT_INCOME_TOTAL": [100000.0, 200000.0, 50000.0],
    "AMT_CREDIT": [500000.0, 400000.0, 250000.0],
    "AMT_ANNUITY": [25000.0, 20000.0, 12500.0],
    "CNT_FAM_MEMBERS": [2.0, 1.0, 4.0],
}


def _write_applications(directory: Path) -> Path:
    train = pd.DataFrame(APPLICATION_COLUMNS)
    test = train.drop(columns=["TARGET"]).head(1).assign(SK_ID_CURR=[9])
    train.to_csv(directory / "application_train.csv", index=False)
    test.to_csv(directory / "application_test.csv", index=False)
    return directory


# --- The encoder -----------------------------------------------------------


def test_one_hot_encoder_creates_dummy_columns_and_names_them() -> None:
    frame = pd.DataFrame({"A": ["x", "y", None], "B": [1, 2, 3]})

    out, new_columns = one_hot_encoder(frame, nan_as_category=True)

    assert "B" in out.columns
    assert any(column.startswith("A_") for column in out.columns)
    assert new_columns and all(column.startswith("A_") for column in new_columns)


def test_the_encoder_keeps_a_column_for_the_missing_category() -> None:
    """`nan_as_category` is what tells « no value » from « the value is absent »."""
    frame = pd.DataFrame({"A": ["x", None]})

    with_nan, _ = one_hot_encoder(frame, nan_as_category=True)
    without_nan, _ = one_hot_encoder(frame, nan_as_category=False)

    assert "A_nan" in with_nan.columns
    assert "A_nan" not in without_nan.columns


# --- The application table -------------------------------------------------


def test_a_missing_raw_table_is_named_rather_than_discovered_later(tmp_path: Path) -> None:
    """The defect it replaces: a matrix a hundred columns short, trained on in silence."""
    with pytest.raises(FileNotFoundError) as refusal:
        _check_raw_files(tmp_path)

    message = str(refusal.value)
    assert all(name in message for name in REQUIRED_RAW_FILES)
    assert str(tmp_path) in message


def test_the_not_employed_sentinel_becomes_missing(tmp_path: Path) -> None:
    """365243 days of employment is a thousand years, and it means « never employed »."""
    frame = application_train_test(_write_applications(tmp_path))

    assert 365243 not in set(frame["DAYS_EMPLOYED"].dropna())
    assert frame["DAYS_EMPLOYED"].isna().sum() == 1


def test_the_applicant_of_unknown_gender_is_dropped(tmp_path: Path) -> None:
    """Four rows of the real table carry XNA, and factorising them invents a third sex."""
    frame = application_train_test(_write_applications(tmp_path))

    assert 3 not in set(frame["SK_ID_CURR"])
    assert set(frame["CODE_GENDER"]) <= {0, 1}


def test_train_and_test_are_concatenated_into_one_frame(tmp_path: Path) -> None:
    """One encoding pass over both, so the two never end up with different columns."""
    frame = application_train_test(_write_applications(tmp_path))

    assert set(frame["SK_ID_CURR"]) == {1, 2, 9}
    assert frame["TARGET"].isna().sum() == 1, "the test rows carry no target"


def test_the_five_ratios_are_computed_from_the_columns_they_name(tmp_path: Path) -> None:
    """Each is a division the model reads; a wrong denominator is invisible downstream."""
    frame = application_train_test(_write_applications(tmp_path)).set_index("SK_ID_CURR")
    applicant = frame.loc[1]

    assert applicant["INCOME_CREDIT_PERC"] == pytest.approx(100000.0 / 500000.0)
    assert applicant["INCOME_PER_PERSON"] == pytest.approx(100000.0 / 2.0)
    assert applicant["ANNUITY_INCOME_PERC"] == pytest.approx(25000.0 / 100000.0)
    assert applicant["PAYMENT_RATE"] == pytest.approx(25000.0 / 500000.0)
    assert applicant["DAYS_EMPLOYED_PERC"] == pytest.approx(-2000 / -12000)


def test_a_ratio_on_the_sentinel_row_is_missing_and_not_infinite(tmp_path: Path) -> None:
    """The row whose employment was a sentinel has no employment ratio to compute."""
    frame = application_train_test(_write_applications(tmp_path)).set_index("SK_ID_CURR")

    value = frame.loc[2, "DAYS_EMPLOYED_PERC"]
    assert np.isnan(value)
