"""The three charts of the log page: their scale, their colours, and their direction.

Each replaces one line of Streamlit's own charting, and each replacement exists for a
reason a test can state. A probability histogram that rescales itself to its data hides the
shift the page is there to show. A decision column of 0 and 1 handed to Plotly as a
number comes back painted as a gradient. And a latency series read newest-first is a chart
that runs backwards.
"""

from __future__ import annotations

import pandas as pd
import pytest

from credexp.app.charts import (
    DECISION_COLOURS,
    decisions_split,
    latency_over_requests,
    probability_distribution,
)
from credexp.figure_style import PALETTE, STATE


@pytest.fixture()
def log() -> pd.DataFrame:
    """Six requests, newest first, as the log page reads them."""
    return pd.DataFrame(
        {
            "created_at": pd.date_range("2026-09-06", periods=6, freq="-1D"),
            "proba_default": [0.9, 0.8, 0.6, 0.4, 0.2, 0.1],
            "decision": [1, 1, 1, 0, 0, 0],
            "latency_ms": [60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )


def test_the_probability_axis_is_the_whole_scale(log) -> None:
    """Fixed from 0 to 1, so two readings of the page are comparable."""
    figure = probability_distribution(log)

    assert tuple(figure.layout.xaxis.range) == (0.0, 1.0)


def test_the_threshold_is_drawn_on_the_distribution(log) -> None:
    """The one number the page exists around, on the axis it applies to."""
    figure = probability_distribution(log, threshold=0.49)

    lines = [shape for shape in figure.layout.shapes if shape.type == "line"]
    assert lines and lines[0].x0 == 0.49
    assert lines[0].line.color == PALETTE["reference"]


def test_the_distribution_is_drawn_without_a_threshold_when_none_is_given(log) -> None:
    figure = probability_distribution(log)

    assert not figure.layout.shapes


def test_the_latency_chart_runs_oldest_to_newest(log) -> None:
    """The log arrives newest first, and a chart that plots it as it comes runs backwards."""
    figure = latency_over_requests(log)

    assert list(figure.data[0].y) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_the_two_outcomes_are_words_before_they_are_colours(log) -> None:
    """`0` and `1` are the database's spelling, and no one reads a bar chart of integers."""
    figure = decisions_split(log)

    labels = set()
    for trace in figure.data:
        labels.update(trace.x)
    assert labels == {"accepted", "refused"}


def test_an_accepted_applicant_is_green_and_a_refused_one_is_red(log) -> None:
    """The two state colours the figures of the README already use for good and bad."""
    assert DECISION_COLOURS == {"accepted": STATE["ok"], "refused": STATE["danger"]}


def test_every_chart_is_painted_on_the_palette_and_not_on_the_library(log) -> None:
    """Nothing of the Streamlit theme reaches Plotly: an unstyled chart arrives in blue."""
    for figure in (
        probability_distribution(log, 0.49),
        latency_over_requests(log),
        decisions_split(log),
    ):
        assert figure.layout.paper_bgcolor == PALETTE["paper"]
        assert figure.layout.font.color == PALETTE["ink"]
        assert figure.layout.xaxis.title.text
        assert figure.layout.yaxis.title.text
