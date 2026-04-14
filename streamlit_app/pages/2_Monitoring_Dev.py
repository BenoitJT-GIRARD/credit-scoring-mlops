import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from credexp.config import settings

st.title("📈 Monitoring Dev")
st.write("Vue simple sur les prédictions stockées en base PostgreSQL.")

engine = create_engine(settings.database_url, pool_pre_ping=True)

query = """
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
        df = pd.read_sql(text(query), conn)

    if df.empty:
        st.warning("Aucune prédiction enregistrée pour le moment.")
        st.stop()

    st.subheader("Dernières prédictions")
    st.dataframe(df, use_container_width=True)

    st.subheader("Indicateurs synthétiques")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nb prédictions", len(df))
    col2.metric("Latence moyenne (ms)", f"{df['latency_ms'].mean():.2f}")
    col3.metric("Score moyen", f"{df['proba_default'].mean():.3f}")

    st.subheader("Distribution des scores")
    st.bar_chart(df["proba_default"])

    st.subheader("Latence")
    st.line_chart(df["latency_ms"].reset_index(drop=True))

    st.subheader("Répartition des décisions")
    decision_counts = df["decision"].value_counts().sort_index()
    st.bar_chart(decision_counts)

except Exception as exc:
    st.error(f"Impossible de lire la base PostgreSQL : {exc}")
