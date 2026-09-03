"""What the service has been deciding, read back from the prediction log."""

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from credexp.config import settings

st.set_page_config(page_title="Credit scoring", page_icon="💳", layout="wide")

st.title("Recent decisions")
st.write(
    "Every call to `/predict` writes a row to PostgreSQL: the score, the decision, the "
    "model version that produced it, and how long it took. This page reads the last 200."
)

engine = create_engine(settings.database_url, pool_pre_ping=True)

QUERY = """
SELECT
    created_at,
    sk_id_curr,
    model_name,
    model_version,
    proba_default,
    decision,
    latency_ms,
    status_code
FROM predictions
ORDER BY created_at DESC
LIMIT 200
"""

try:
    with engine.connect() as conn:
        df = pd.read_sql(text(QUERY), conn)

    if df.empty:
        st.warning("The log is empty — nothing has been scored yet.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Decisions logged", len(df))
    col2.metric("Mean latency", f"{df['latency_ms'].mean():.1f} ms")
    col3.metric("Mean probability", f"{df['proba_default'].mean():.3f}")

    st.subheader("The log")
    st.dataframe(df, use_container_width=True)

    st.subheader("Distribution of the predicted probability")
    st.caption("The earliest warning available: it moves before any label arrives.")
    st.bar_chart(df["proba_default"])

    st.subheader("Latency, request by request")
    st.line_chart(df["latency_ms"].reset_index(drop=True))

    st.subheader("How the threshold split the traffic")
    st.caption("0 accepted, 1 refused. The acceptance rate is what the business side watches.")
    st.bar_chart(df["decision"].value_counts().sort_index())

except Exception as exc:  # noqa: BLE001 - the page must say why it is empty, not traceback
    st.error(f"Could not read the prediction log from PostgreSQL — {exc}")
