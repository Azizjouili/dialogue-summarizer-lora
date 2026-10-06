FROM python:3.11-slim

ENV UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=2 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen || uv sync --no-dev

COPY model.py api.py ./
COPY static ./static

EXPOSE 8000
CMD [".venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]