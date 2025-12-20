FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    curl -L --fail "${BASE_URL}/avm_catboost_multihot187.cbm" -o "/app/artifacts/${MODEL_VERSION}/avm_catboost_multihot187.cbm"; \
    curl -L --fail "${BASE_URL}/microlocation_vocab.json"     -o "/app/artifacts/${MODEL_VERSION}/microlocation_vocab.json"; \
    curl -L --fail "${BASE_URL}/training_schema.json"         -o "/app/artifacts/${MODEL_VERSION}/training_schema.json"; \
    ls -lah "/app/artifacts/${MODEL_VERSION}"

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
