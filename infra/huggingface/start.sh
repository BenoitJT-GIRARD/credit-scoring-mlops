#!/usr/bin/env bash
set -e

# `--no-sync` on every `uv run`: the image installed its groups at build time, and a plain
# `uv run` would add the default `dev` group at each start, over the network.

echo "Starting Credit Scoring MLOps Hugging Face Space"

echo "Checking model artifacts..."
test -f /app/models/pipeline.joblib
test -f /app/models/threshold.json
test -f /app/models/feature_columns.json

echo "Model artifacts found."

if [ -n "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL is configured. Initializing remote database..."
  uv run --no-sync python scripts/init_db.py || echo "Database initialization failed, continuing without blocking API startup."
else
  echo "DATABASE_URL is not configured. Database logging will be skipped or fail best-effort."
fi

echo "Starting FastAPI on 127.0.0.1:8000"
uv run --no-sync uvicorn credexp.serving.api:app \
  --host 127.0.0.1 \
  --port 8000 &

echo "Starting Streamlit on 127.0.0.1:8501"
uv run --no-sync streamlit run src/credexp/app/Overview.py \
  --server.port=8501 \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --browser.gatherUsageStats=false &

echo "Starting Nginx on public port 7860"
nginx -g "daemon off;"
