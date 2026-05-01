FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY streamlit_app ./streamlit_app

RUN uv sync --no-dev --group serve --group db --group monitoring --group mlops --group ml --group data --group perf

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "-c", "import credexp; print('CI image build OK')"]
