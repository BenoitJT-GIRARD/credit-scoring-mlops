"""The log query, and the three numbers above it.

The query runs against SQLite here, which is not what production reads. What is under test
is not the driver: it is that the page reads a bounded, ordered slice, and that an empty
log produces zeros instead of a NaN in a metric card.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from credexp.app.prediction_log import COLUMNS, DEFAULT_LIMIT, recent_decisions, summarise


@pytest.fixture()
def log():
    """Twelve scored requests, in a database that lives in this process and nowhere else."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE predictions ("
                "created_at TEXT, sk_id_curr INTEGER, model_name TEXT, model_version TEXT,"
                "proba_default REAL, decision INTEGER, latency_ms REAL, status_code INTEGER)"
            )
        )
        for index in range(12):
            connection.execute(
                text(
                    "INSERT INTO predictions VALUES (:created_at, :sk_id_curr, 'm', 'v',"
                    " :proba, :decision, :latency, 200)"
                ),
                {
                    "created_at": f"2026-09-{index + 1:02d}T00:00:00",
                    "sk_id_curr": index,
                    "proba": index / 12,
                    "decision": int(index >= 6),
                    "latency": 10.0 + index,
                },
            )
    return engine


def test_the_page_reads_the_columns_it_shows(log) -> None:
    frame = recent_decisions(log)

    assert list(frame.columns) == list(COLUMNS)


def test_the_most_recent_request_comes_first(log) -> None:
    """« Recent » ordered by nothing is « whichever rows the planner reached first »."""
    frame = recent_decisions(log)

    assert frame["created_at"].to_list() == sorted(frame["created_at"], reverse=True)


def test_the_query_is_bounded(log) -> None:
    """The table grows with every request the service answers; the page does not."""
    frame = recent_decisions(log, limit=5)

    assert len(frame) == 5
    assert frame["sk_id_curr"].to_list() == [11, 10, 9, 8, 7]


def test_the_default_bound_is_the_one_the_page_announces() -> None:
    """The page writes « the last 200 » into its own sentence, from this constant."""
    assert DEFAULT_LIMIT == 200


def test_the_three_numbers_describe_the_slice_that_was_read(log) -> None:
    frame = recent_decisions(log, limit=4)

    summary = summarise(frame)

    assert summary["n"] == 4
    assert summary["mean_latency_ms"] == pytest.approx(frame["latency_ms"].mean())
    assert summary["mean_probability"] == pytest.approx(frame["proba_default"].mean())
    assert summary["refusal_rate"] == 1.0, "the four most recent are all above the threshold"


def test_an_empty_log_summarises_to_zeros_and_not_to_nan() -> None:
    """A NaN in a metric card is a card that says nothing, where « 0 » says what happened."""
    summary = summarise(pd.DataFrame(columns=list(COLUMNS)))

    assert summary == {"n": 0, "mean_latency_ms": 0.0, "mean_probability": 0.0, "refusal_rate": 0.0}
