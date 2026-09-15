"""Landing page of the demonstration interface.

The two pages under `pages/` are the two things a reader usually wants to see without
installing anything: what the API answers for one applicant, and what the service has been
answering lately.
"""

import streamlit as st

st.set_page_config(
    page_title="Credit scoring — demonstration interface",
    page_icon="💳",
    layout="wide",
)

st.title("Credit scoring")
st.markdown(
    """
A model that estimates the probability an applicant defaults, and a threshold that turns
that probability into an accept-or-refuse decision.

**Score an applicant** sends one applicant to the API and shows what comes back:
the probability, the decision, and how long it took.

**Recent decisions** reads the prediction log out of PostgreSQL — every score the service
has produced, with the model version that produced it.

The threshold is not a rounding of 0.5. It is chosen on a validation split to minimise an
expected cost in which refusing a good applicant and accepting a defaulting one are not
worth the same thing.
"""
)

st.caption("FastAPI for inference · PostgreSQL for the decision log · Streamlit for this page")
st.info("Use the sidebar to move between the two pages.")
