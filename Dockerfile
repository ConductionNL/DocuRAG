# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.8.20 AS uvbin

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    HF_HOME=/tmp/.cache/huggingface \
    HOME=/tmp \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=80 \
    WEB_CONCURRENCY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv by copying from the official image (faster and more reliable than curl during builds)
COPY --from=uvbin /uv /usr/local/bin/uv

# Copy project metadata (and lock if present), then install with uv sync
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv bash -c "uv sync || (echo 'uv sync failed, retrying in 10s' && sleep 10 && uv sync)"

COPY . .

EXPOSE 80

# Create non-root user and adjust permissions
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app && \
    mkdir -p ${HF_HOME} && chown -R appuser:appuser ${HF_HOME}

USER appuser

# Run uvicorn with configurable workers (via uv)
# Ensure cache dirs are writable and unique per UID to avoid permission issues across restarts
CMD ["sh", "-c", "uid=$(id -u); export XDG_CACHE_HOME=/tmp/uvcache-$uid UV_CACHE_DIR=\"$XDG_CACHE_HOME/uv\" HF_HOME=\"$XDG_CACHE_HOME/huggingface\" HOME=/tmp; mkdir -p \"$UV_CACHE_DIR\" \"$HF_HOME\" && chmod -R 0777 \"$XDG_CACHE_HOME\" && uv run -- uvicorn app:app --host ${UVICORN_HOST} --port ${UVICORN_PORT} --workers ${WEB_CONCURRENCY}"]