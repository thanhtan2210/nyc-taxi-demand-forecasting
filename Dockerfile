# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install system dependencies needed for compiling some packages or running ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flake8

# Copy application source code and models
COPY app/ ./app/
COPY src/ ./src/
COPY sql/ ./sql/
COPY tests/ ./tests/
COPY saved_models/ ./saved_models/
COPY dataset/taxi_zone_lookup.csv ./dataset/taxi_zone_lookup.csv

# Expose Streamlit default port
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
