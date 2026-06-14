FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv venv /app/.venv && \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

COPY app.py .
COPY frontend/ frontend/

VOLUME ["/app/chroma_db"]
EXPOSE 5000
CMD ["python", "app.py"]