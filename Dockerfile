# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    WEB_CONCURRENCY=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80

# Create non-root user and adjust permissions
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app && chown -R appuser:appuser /app && \
    mkdir -p ${HF_HOME} && chown -R appuser:appuser ${HF_HOME}

USER appuser

# Run uvicorn with configurable workers
CMD ["sh", "-c", "uvicorn app:app --host ${UVICORN_HOST} --port ${UVICORN_PORT} --workers ${WEB_CONCURRENCY}"]