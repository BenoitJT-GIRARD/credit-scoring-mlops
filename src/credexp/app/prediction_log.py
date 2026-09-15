"""Reading back what the service has been deciding, and summarising it in three numbers.

The query lives here rather than in the page for the same reason the API call does: a
`SELECT` inside a Streamlit script can only be checked by opening a browser, and the two
things worth checking about this one are that it is bounded and that it is ordered. A log
page that reads the whole table grows slower every day it is used, and one that does not
order by time shows « recent » rows chosen by the planner.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from credexp.config import settings

#: The columns the page shows, in the order it shows them.
COLUMNS = (
    "created_at",
    "sk_id_curr",
    "model_name",
    "model_version",
    "proba_default",
    "decision",
    "latency_ms",
    "status_code",
)

#: How many rows the page reads. Bounded on purpose: this is a demonstration page over a
#: table that grows with every request.
DEFAULT_LIMIT = 200

QUERY = text(f"SELECT {', '.join(COLUMNS)} FROM predictions ORDER BY created_at DESC LIMIT :limit")


def log_engine() -> Engine:
    """The engine the page reads through, built from the configured database URL.

    Here rather than in the page for the same reason as the query: a connection opened in
    a Streamlit script is a connection no test can look at.
    """
    return create_engine(settings.database_url, pool_pre_ping=True)


def recent_decisions(engine: Engine, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """The last `limit` scored requests, most recent first."""
    with engine.connect() as connection:
        return pd.read_sql(QUERY, connection, params={"limit": limit})


def summarise(frame: pd.DataFrame) -> dict[str, float | int]:
    """The three numbers above the table: how many, how fast, how risky on average.

    An empty log answers with zeros rather than with a NaN: the page says « nothing has
    been scored yet » and a NaN in a metric card says nothing at all.
    """
    if frame.empty:
        return {"n": 0, "mean_latency_ms": 0.0, "mean_probability": 0.0, "refusal_rate": 0.0}
    return {
        "n": int(len(frame)),
        "mean_latency_ms": float(frame["latency_ms"].mean()),
        "mean_probability": float(frame["proba_default"].mean()),
        "refusal_rate": float(frame["decision"].mean()),
    }
