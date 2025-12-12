FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

# Create data directory
RUN mkdir -p data

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/orchestrate.py", "status"]
