FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* README.md ./
COPY src ./src

RUN uv sync --no-dev --group data --group db --group ml --group mlops --group monitoring --group serve

COPY scripts ./scripts
COPY data/processed ./data/processed
COPY artifacts/models ./artifacts/models
COPY mlflow ./mlflow

ENV PYTHONPATH=/app/src
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

CMD ["uv", "run", "python", "scripts/run_api.py"]