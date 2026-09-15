"""Aligning two populations before comparing them, and typing the columns for Evidently.

Drift is measured on what callers send, and a caller sends what it has. The reference has
everything. Comparing them without an alignment step compares a column against a column
that does not exist, and Evidently reports that as drift — a monitoring page that turns red
because of a payload shape rather than because of a population.

Two columns are dropped by name. `TARGET` is an outcome and is never in an API payload;
`created_at` is the row's own timestamp and would drift by construction, every day, for
ever.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credexp.monitoring.drift import (
    _sanitize_dataframe,
    align_reference_and_current,
    infer_data_definition,
    save_drift_report,
)


@pytest.fixture()
def reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.1, 0.2, 0.3],
            "AMT_CREDIT": [1000.0, 2000.0, 3000.0],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving"],
            "TARGET": [0, 1, 0],
        }
    )


# --- The alignment ----------------------------------------------------------


def test_only_the_columns_both_sides_carry_are_compared(reference) -> None:
    current = pd.DataFrame({"EXT_SOURCE_1": [0.9], "SOMETHING_NEW": [1.0]})

    ref, cur = align_reference_and_current(reference, current)

    assert list(ref.columns) == ["EXT_SOURCE_1"]
    assert list(cur.columns) == ["EXT_SOURCE_1"]


def test_the_outcome_column_is_never_part_of_the_comparison(reference) -> None:
    """`TARGET` is what the model predicts, and no API payload carries it."""
    current = reference.copy()

    ref, cur = align_reference_and_current(reference, current)

    assert "TARGET" not in ref.columns
    assert "TARGET" not in cur.columns


def test_the_timestamp_is_dropped_because_it_drifts_by_construction(reference) -> None:
    current = reference.drop(columns=["TARGET"]).assign(
        created_at=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    )

    _, cur = align_reference_and_current(reference, current)

    assert "created_at" not in cur.columns


def test_two_populations_with_nothing_in_common_are_refused(reference) -> None:
    """Silence here would be a monitoring page comparing an empty frame to an empty frame."""
    with pytest.raises(ValueError, match="No common columns"):
        align_reference_and_current(reference, pd.DataFrame({"UNRELATED": [1.0]}))


def test_the_columns_come_back_in_the_same_order_on_both_sides(reference) -> None:
    """Evidently pairs them positionally as much as by name, so the order is the contract."""
    current = pd.DataFrame(
        {"AMT_CREDIT": [5.0], "NAME_CONTRACT_TYPE": ["Cash"], "EXT_SOURCE_1": [0.5]}
    )

    ref, cur = align_reference_and_current(reference, current)

    assert list(ref.columns) == list(cur.columns)


# --- The typing -------------------------------------------------------------


def test_a_text_column_is_categorical_and_a_number_is_not(reference) -> None:
    definition = infer_data_definition(reference.drop(columns=["TARGET"]))

    assert set(definition.numerical_columns) == {"EXT_SOURCE_1", "AMT_CREDIT"}
    assert definition.categorical_columns == ["NAME_CONTRACT_TYPE"]


def test_a_boolean_column_is_read_as_a_category() -> None:
    """A flag with two values is not a quantity, and a drift test on it is a proportion test."""
    definition = infer_data_definition(pd.DataFrame({"FLAG_OWN_CAR": [True, False]}))

    assert definition.categorical_columns == ["FLAG_OWN_CAR"]


# --- Infinities -------------------------------------------------------------


def test_an_infinity_is_read_as_missing_before_anything_is_compared() -> None:
    """A ratio with a zero denominator would otherwise be one end of every histogram."""
    frame = pd.DataFrame({"ratio": [1.0, np.inf, -np.inf]})

    out = _sanitize_dataframe(frame)

    assert out["ratio"].isna().sum() == 2
    assert frame["ratio"].isna().sum() == 0, "the caller's frame is left alone"


# --- What the run writes ----------------------------------------------------


class _Report:
    """Stands in for an Evidently report, which needs two real populations to build.

    Building one is the integration tier's business. What is asserted here is the metadata
    file written beside it, which is the part `reports/monitoring/` tracks.
    """

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save_html(self, path: str) -> None:
        self.saved_to = path
        Path(path).write_text("<html></html>", encoding="utf-8")


def test_the_metadata_names_the_two_populations_and_the_report(tmp_path: Path) -> None:
    report = _Report()
    html = tmp_path / "nested" / "drift.html"
    meta = tmp_path / "nested" / "drift_meta.json"

    save_drift_report(report, html, meta, n_reference=1000, n_current=300, n_features=42)

    recorded = json.loads(meta.read_text(encoding="utf-8"))
    assert recorded["n_reference"] == 1000
    assert recorded["n_current"] == 300
    assert recorded["n_features"] == 42
    assert recorded["report_path"] == str(html)
    assert html.is_file()


def test_the_report_can_be_written_without_its_metadata(tmp_path: Path) -> None:
    report = _Report()
    html = tmp_path / "drift.html"

    save_drift_report(report, html, None, n_reference=1, n_current=1, n_features=1)

    assert html.is_file()
    assert not list(tmp_path.glob("*.json"))
