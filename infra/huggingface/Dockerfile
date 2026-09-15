FROM python:3.12-slim

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    nginx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.6

COPY pyproject.toml uv.lock /app/
COPY README.md /app/README.md
COPY src /app/src
COPY scripts /app/scripts
COPY src/credexp/app /app/src/credexp/app
COPY models /app/models

COPY infra/huggingface/nginx.conf /etc/nginx/conf.d/default.conf
COPY infra/huggingface/start.sh /app/start.sh

# Important on Windows: remove CRLF if Git converted line endings.
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

RUN uv sync --frozen \
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
ENV MODEL_JOBLIB_PATH=/app/models/pipeline.joblib
ENV THRESHOLD_PATH=/app/models/threshold.json
ENV FEATURE_COLUMNS_PATH=/app/models/feature_columns.json

# Internal services.
ENV API_HOST=127.0.0.1
ENV API_PORT=8000
ENV API_ROOT_PATH=/api
ENV API_BASE_URL=http://127.0.0.1:8000

# Metrics are available through /api/metrics.
ENV PROMETHEUS_ENABLED=true

EXPOSE 7860

# Hugging Face Spaces runs containers as uid 1000, and nothing here needs root. nginx keeps
# its pid under /tmp because /run belongs to root, and the three directories it writes to are
# handed to the same user.
RUN useradd --create-home --uid 1000 appuser \
    && sed -i 's#^pid .*#pid /tmp/nginx.pid;#' /etc/nginx/nginx.conf \
    && chown -R appuser:appuser /app /var/log/nginx /var/lib/nginx /etc/nginx/conf.d
USER appuser

# The probe hits the public port, the one the Space itself answers on. Without it, a
# container whose nginx never came up would still be reported as running.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)"

CMD ["/app/start.sh"]
