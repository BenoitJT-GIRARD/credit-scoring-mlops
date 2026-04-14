import streamlit as st

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Scoring Dashboard")
st.markdown(
    """
Bienvenue dans l'interface de démonstration du modèle de scoring crédit.

### Contenu
- **Scoring Client** : permet d’envoyer un client à l’API et de récupérer son score.
- **Monitoring Dev** : permet de visualiser quelques métriques simples de monitoring.

### Architecture
- **FastAPI** pour l’inférence
- **PostgreSQL** pour stocker les prédictions
- **Streamlit** pour l’interface
"""
)

st.info("Utilise le menu latéral pour naviguer entre les pages.")
