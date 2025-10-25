# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY tiktik_rag ./tiktik_rag

EXPOSE 8000

CMD ["uvicorn", "tiktik_rag.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

