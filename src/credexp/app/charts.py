"""The three charts of the log page, drawn in the palette the figures of the README use.

`st.bar_chart` and `st.line_chart` are one line each and paint in Streamlit's own colours,
next to a page painted in this project's. They also decide the binning themselves, which
on a probability between 0 and 1 is the one thing a reader of this page needs to control:
the distribution is the earliest warning available, and it is only readable on a fixed
scale.

Colours come from :mod:`credexp.figure_style`, by name, through this module. No chart of
the interface picks one of its own.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px

from credexp.figure_style import PALETTE, STATE

#: Accepted and refused, in the two state colours the figures already use for good and bad.
DECISION_COLOURS = {"accepted": STATE["ok"], "refused": STATE["danger"]}

LAYOUT = {
    "paper_bgcolor": PALETTE["paper"],
    "plot_bgcolor": PALETTE["paper"],
    "font": {"color": PALETTE["ink"], "size": 13},
    "margin": {"t": 40, "b": 40, "l": 10, "r": 10},
    "showlegend": False,
}
AXIS = {
    "gridcolor": PALETTE["grid"],
    "zerolinecolor": PALETTE["grid"],
    "linecolor": PALETTE["muted"],
}


def styled(figure: Any, x_title: str, y_title: str) -> Any:
    """One place where a chart of the interface gets its colours and its axis titles."""
    figure.update_layout(**LAYOUT)
    figure.update_xaxes(title_text=x_title, **AXIS)
    figure.update_yaxes(title_text=y_title, **AXIS)
    return figure


def probability_distribution(frame: pd.DataFrame, threshold: float | None = None) -> Any:
    """Where the scores fall, on the full 0-to-1 scale, with the threshold drawn on it.

    The scale is fixed rather than fitted to the data. A histogram that rescales itself
    hides exactly the shift this page exists to show.
    """
    figure = px.histogram(
        frame,
        x="proba_default",
        nbins=40,
        range_x=[0.0, 1.0],
        color_discrete_sequence=[PALETTE["primary"]],
    )
    if threshold is not None:
        figure.add_vline(
            x=threshold,
            line_dash="dot",
            line_color=PALETTE["reference"],
            annotation_text=f"threshold {threshold:g}",
        )
    return styled(figure, "Probability of default", "Requests")


def latency_over_requests(frame: pd.DataFrame) -> Any:
    """Latency in the order the requests arrived, oldest first."""
    ordered = frame.iloc[::-1].reset_index(drop=True)
    figure = px.line(ordered, y="latency_ms", color_discrete_sequence=[PALETTE["primary"]])
    return styled(figure, "Request, oldest first", "Latency (ms)")


def decisions_split(frame: pd.DataFrame) -> Any:
    """How the threshold split the traffic, as two named bars rather than 0 and 1.

    A column of 0 and 1 arrives at Plotly as a quantity, and two values get a gradient
    where they needed two names.
    """
    counts = (
        frame["decision"]
        .map({0: "accepted", 1: "refused"})
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="requests")
    )
    figure = px.bar(
        counts,
        x="outcome",
        y="requests",
        color="outcome",
        color_discrete_map=DECISION_COLOURS,
    )
    return styled(figure, "What the threshold decided", "Requests")
