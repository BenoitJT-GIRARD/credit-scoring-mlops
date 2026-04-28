import json
import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.title("🧮 Scoring Client")
st.write("Envoie une requête à l’API FastAPI pour obtenir un score de défaut.")

default_payload = {
    "sk_id_curr": 123456,
    "features": {
        "EXT_SOURCE_1": 0.52,
        "EXT_SOURCE_2": 0.71,
        "EXT_SOURCE_3": 0.41,
    },
}

payload_text = st.text_area(
    "Payload JSON",
    value=json.dumps(default_payload, indent=2),
    height=250,
)

st.caption(f"API utilisée : {API_BASE_URL}")

if st.button("Scorer ce client"):
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        st.error(f"JSON invalide : {exc}")
        st.stop()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{API_BASE_URL}/predict", json=payload)
        response.raise_for_status()
        data = response.json()

        st.success("Prédiction réussie")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probabilité de défaut", f"{data['proba_default']:.3f}")
        col2.metric("Décision", str(data["decision"]))
        col3.metric("Latence (ms)", f"{data['latency_ms']:.2f}")

        st.subheader("Réponse complète")
        st.json(data)

    except httpx.HTTPStatusError as exc:
        st.error(f"Erreur HTTP {exc.response.status_code}")
        st.json(exc.response.json())
    except Exception as exc:
        st.error(f"Erreur : {exc}")
