FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.6

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app
COPY artifacts/models ./artifacts/models

RUN uv sync --frozen \
    --no-dev \
    --group serve \
    --group db \
    --group monitoring \
    --group mlops \
    --group ml \
    --group data

ENV PYTHONPATH=/app/src

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

ENV MODEL_LOAD_MODE=joblib
ENV MODEL_NAME=credit_scoring_model
ENV MODEL_VERSION=local-joblib
ENV MODEL_JOBLIB_PATH=/app/artifacts/models/pipeline.joblib
ENV THRESHOLD_PATH=/app/artifacts/models/threshold.json
ENV FEATURE_COLUMNS_PATH=/app/artifacts/models/feature_columns.json

CMD ["uv", "run", "python", "scripts/run_api.py"]
