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
COPY src/credexp/app ./src/credexp/app
COPY models ./models
# The interface runs from this image too, and Streamlit reads its theme and toolbar
# settings from `.streamlit/` in the working directory.
COPY .streamlit ./.streamlit

# No `--group mlops`. The registry is a training-time convenience, and the loader imports
# MLflow only on the path that resolves a registered version, which this image is
# configured never to take. Leaving the group out takes hundreds of megabytes with it.
RUN uv sync --frozen \
    --no-dev \
    --group serve \
    --group db \
    --group monitoring \
    --group ml \
    --group data

ENV PYTHONPATH=/app/src

ENV API_HOST=0.0.0.0
ENV API_PORT=8000

ENV MODEL_LOAD_MODE=joblib
ENV MODEL_NAME=credit_scoring_model
ENV MODEL_VERSION=local-joblib
ENV MODEL_JOBLIB_PATH=/app/models/pipeline.joblib
ENV THRESHOLD_PATH=/app/models/threshold.json
ENV FEATURE_COLUMNS_PATH=/app/models/feature_columns.json

EXPOSE 8000

# The liveness probe is the API's own `/health` route. It does not read the model, and it
# answers once the startup, which loads the model, has finished.
# The dashboard service builds from this same image and overrides the probe in the compose:
# it listens on another port, and the inherited one would report it dead forever.
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Nothing here needs root.
RUN useradd --create-home --uid 1000 credexp && chown -R credexp:credexp /app
USER credexp

# `--no-sync` starts the environment installed above, unchanged. A plain `uv run` would first
# install the default `dev` group, which the build leaves out, so every start would need the
# network and would download the test and lint tools.
CMD ["uv", "run", "--no-sync", "python", "scripts/run_api.py"]
