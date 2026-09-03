"""Send one applicant to the API and show what comes back."""

import json
import os

import httpx
import streamlit as st

st.set_page_config(page_title="Credit scoring", page_icon="💳", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

#: Three external credit-bureau scores. They carry most of the signal in this model, and
#: any feature left out of the payload is imputed, so a short request is a valid one.
DEFAULT_PAYLOAD = {
    "sk_id_curr": 123456,
    "features": {
        "EXT_SOURCE_1": 0.52,
        "EXT_SOURCE_2": 0.71,
        "EXT_SOURCE_3": 0.41,
    },
}

st.title("Score an applicant")
st.write(
    "Edit the payload and send it to `POST /predict`. Features you leave out are imputed "
    "by the pipeline, so a three-feature request is enough to get a score."
)

payload_text = st.text_area("Request body", value=json.dumps(DEFAULT_PAYLOAD, indent=2), height=250)
st.caption(f"API: {API_BASE_URL}")

if st.button("Score this applicant", type="primary"):
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        st.error(f"That is not valid JSON — {exc}")
        st.stop()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{API_BASE_URL}/predict", json=payload)
        response.raise_for_status()
        data = response.json()

        st.success("Scored, and written to the prediction log.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of default", f"{data['proba_default']:.3f}")
        col2.metric("Decision", "refuse" if data["decision"] else "accept")
        col3.metric("Latency", f"{data['latency_ms']:.1f} ms")

        st.subheader("Full response")
        st.json(data)

    except httpx.HTTPStatusError as exc:
        st.error(f"The API refused the request — HTTP {exc.response.status_code}")
        st.json(exc.response.json())
    except httpx.HTTPError as exc:
        st.error(f"The API did not answer — {exc}. Is it running at {API_BASE_URL}?")
