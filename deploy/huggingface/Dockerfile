FROM python:3.12-slim

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    nginx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* /app/
COPY README.md /app/README.md
COPY src /app/src
COPY scripts /app/scripts
COPY streamlit_app /app/streamlit_app
COPY artifacts/models /app/artifacts/models

COPY deploy/huggingface/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/huggingface/start.sh /app/start.sh

# Important on Windows: remove CRLF if Git converted line endings.
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

RUN uv sync \
    --no-dev \
    --group data \
    --group ml \
    --group mlops \
    --group serve \
    --group db \
    --group monitoring

ENV PYTHONPATH=/app/src

# Model loading: remote joblib mode.
ENV MODEL_LOAD_MODE=joblib
ENV MODEL_NAME=credit_scoring_model
ENV MODEL_VERSION=hf-space
ENV MODEL_JOBLIB_PATH=/app/artifacts/models/pipeline.joblib
ENV THRESHOLD_PATH=/app/artifacts/models/threshold.json
ENV FEATURE_COLUMNS_PATH=/app/artifacts/models/feature_columns.json

# Internal services.
ENV API_HOST=127.0.0.1
ENV API_PORT=8000
ENV API_ROOT_PATH=/api
ENV API_BASE_URL=http://127.0.0.1:8000

# Metrics are available through /api/metrics.
ENV PROMETHEUS_ENABLED=true

EXPOSE 7860

CMD ["/app/start.sh"]
