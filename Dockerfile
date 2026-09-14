FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY machineguard/ machineguard/
COPY scripts/ scripts/

# Create directories for models and data
RUN mkdir -p models/artifacts data

# Expose API port
EXPOSE 8000

# Default command: run the API server
CMD ["uvicorn", "machineguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
