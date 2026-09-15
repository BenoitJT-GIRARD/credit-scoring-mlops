"""Send one applicant to the API and show what comes back.

The page is the screen and nothing else: reading the payload, calling the service and
naming the three ways that can fail live in `credexp.app.service`, where a test can reach
them without a browser.
"""

import json

import streamlit as st

from credexp.app.service import DEFAULT_BASE_URL, EXAMPLE_PAYLOAD, parse_payload, score

st.set_page_config(page_title="Credit scoring", page_icon="💳", layout="wide")

st.title("Score an applicant")
st.write(
    "Edit the payload and send it to `POST /predict`. Features you leave out are imputed "
    "by the pipeline, so a three-feature request is enough to get a score."
)

payload_text = st.text_area("Request body", value=json.dumps(EXAMPLE_PAYLOAD, indent=2), height=250)
st.caption(f"API: {DEFAULT_BASE_URL}")

if st.button("Score this applicant", type="primary"):
    payload, syntax_error = parse_payload(payload_text)
    if syntax_error is not None:
        st.error(syntax_error)
        st.stop()

    result = score(payload)
    if not result.ok:
        st.error(result.problem)
        if result.body is not None:
            st.json(result.body)
        st.stop()

    data = result.body
    st.success("Scored, and written to the prediction log.")
    left, middle, right = st.columns(3)
    left.metric("Probability of default", f"{data['proba_default']:.3f}")
    middle.metric("Decision", "refuse" if data["decision"] else "accept")
    right.metric("Latency", f"{data['latency_ms']:.1f} ms")
    left.caption(f"threshold {data['threshold']:g}")
    middle.caption("refuse when the probability is at or above the threshold")
    right.caption(f"model {data['model_name']} {data['model_version']}")

    st.subheader("Full response")
    st.json(data)
