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

EXPOSE 8000

# The liveness probe is the API's own route, the one that answers without the model loaded.
# The dashboard service builds from this same image and overrides the probe in the compose:
# it listens on another port, and the inherited one would report it dead forever.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Nothing here needs root.
RUN useradd --create-home --uid 1000 credexp && chown -R credexp:credexp /app
USER credexp

CMD ["uv", "run", "python", "scripts/run_api.py"]
