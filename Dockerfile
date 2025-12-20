FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

# -----------------------------
# Download artifacts from GitHub Release
# -----------------------------
ARG MODEL_VERSION="v2_multihot187"
ARG BASE_URL="https://github.com/cillegio8/ai-bina-avm-api/releases/download/${MODEL_VERSION}"
ENV MODEL_VERSION=${MODEL_VERSION}

RUN set -eux; \
    mkdir -p "/app/artifacts/${MODEL_VERSION}"; \
    echo "Downloading artifacts from: ${BASE_URL}"; \
    curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 10 \
      "${BASE_URL}/training_schema.json" \
      -o "/app/artifacts/${MODEL_VERSION}/training_schema.json"; \
    curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 10 \
      "${BASE_URL}/microlocation_vocab.json" \
      -o "/app/artifacts/${MODEL_VERSION}/microlocation_vocab.json"; \
    curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 10 \
      "${BASE_URL}/avm_catboost_multihot187.cbm" \
      -o "/app/artifacts/${MODEL_VERSION}/avm_catboost_multihot187.cbm"; \
    ls -lah "/app/artifacts/${MODEL_VERSION}"; \
    test -s "/app/artifacts/${MODEL_VERSION}/training_schema.json"; \
    test -s "/app/artifacts/${MODEL_VERSION}/microlocation_vocab.json"; \
    test -s "/app/artifacts/${MODEL_VERSION}/avm_catboost_multihot187.cbm"

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
