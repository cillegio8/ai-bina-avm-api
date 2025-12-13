FROM python:3.12-slim

# Install system deps + curl (needed to download model)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY main.py .

# -----------------------------
# Download model from GitHub Release
# -----------------------------
ARG MODEL_URL="https://github.com/cillegio8/ai-bina-avm-api/releases/download/v0.1.0/ai_bina_catboost_avm.cbm"

RUN echo "Downloading model from: $MODEL_URL" \
 && curl -L --fail "$MODEL_URL" -o /app/ai_bina_catboost_avm.cbm \
 && ls -lh /app/ai_bina_catboost_avm.cbm

# Expose port
EXPOSE 8080

# Uvicorn command (Northflank-compatible)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
