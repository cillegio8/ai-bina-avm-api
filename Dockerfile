FROM python:3.12-slim

# Install system deps (for CatBoost / Pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + model
COPY main.py .
COPY ai_bina_catboost_avm.cbm .

# Expose port
EXPOSE 8080

# Uvicorn command for DO App Platform (must listen on 0.0.0.0:8080)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
