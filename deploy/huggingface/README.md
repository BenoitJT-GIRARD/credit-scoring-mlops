---
title: Credit Scoring MLOps Demo
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Credit Scoring MLOps Demo

This Hugging Face Space deploys a single Docker container with:

- Streamlit user interface
- FastAPI scoring API
- Nginx reverse proxy
- Supabase PostgreSQL logging

Routes:

- `/` → Streamlit UI
- `/api/docs` → FastAPI Swagger
- `/api/health` → API healthcheck

The full local monitoring stack with PostgreSQL, Prometheus and Grafana is documented in the GitHub repository.
