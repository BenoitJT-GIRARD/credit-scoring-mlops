FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app
COPY data/processed ./data/processed
COPY artifacts/models ./artifacts/models
COPY mlflow ./mlflow

RUN uv sync --no-dev --group serve --group db --group monitoring --group mlops --group ml --group data

ENV PYTHONPATH=/app/src
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["uv", "run", "python", "scripts/run_api.py"]