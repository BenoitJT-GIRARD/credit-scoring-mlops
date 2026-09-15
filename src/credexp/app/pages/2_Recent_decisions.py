"""What the service has been deciding, read back from the prediction log.

The query, the summary and the three charts live in `credexp.app.prediction_log` and
`credexp.app.charts`. What is left here is the screen: four cards, a table, and the three
figures in the palette the README's figures use.
"""

import json

import streamlit as st

from credexp.app.charts import decisions_split, latency_over_requests, probability_distribution
from credexp.app.prediction_log import DEFAULT_LIMIT, log_engine, recent_decisions, summarise
from credexp.utils import THRESHOLD_PATH

st.set_page_config(page_title="Credit scoring", page_icon="💳", layout="wide")

st.title("Recent decisions")
st.write(
    "Every call to `/predict` writes a row to PostgreSQL: the score, the decision, the "
    f"model version that produced it, and how long it took. This page reads the last "
    f"{DEFAULT_LIMIT}."
)

engine = log_engine()
threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"]

try:
    frame = recent_decisions(engine)
except Exception as exc:  # noqa: BLE001 - the page says why it is empty rather than tracing
    st.error(f"Could not read the prediction log from PostgreSQL — {exc}")
    st.stop()

if frame.empty:
    st.warning("The log is empty — nothing has been scored yet.")
    st.stop()

summary = summarise(frame)
cards = st.columns(4)
cards[0].metric("Decisions logged", summary["n"])
cards[0].caption(f"the last {DEFAULT_LIMIT} at most")
cards[1].metric("Mean latency", f"{summary['mean_latency_ms']:.1f} ms")
cards[1].caption("one applicant per request")
cards[2].metric("Mean probability", f"{summary['mean_probability']:.3f}")
cards[2].caption("a ranking score, not a calibrated probability")
cards[3].metric("Refusal rate", f"{summary['refusal_rate']:.1%}")
cards[3].caption(f"at the shipped threshold of {threshold:g}")

st.subheader("The log")
st.dataframe(frame, width="stretch", height=320)

st.subheader("Distribution of the predicted probability")
st.caption("The earliest warning available: it moves before any label arrives.")
st.plotly_chart(probability_distribution(frame, threshold), width="stretch")

st.subheader("Latency, request by request")
st.plotly_chart(latency_over_requests(frame), width="stretch")

st.subheader("How the threshold split the traffic")
st.caption("The acceptance rate is what the business side watches.")
st.plotly_chart(decisions_split(frame), width="stretch")
