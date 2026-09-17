# ── Dockerfile ─────────────────────────────────────────────────
# AI-Powered Customer Support Intelligence Platform
# GUVI × HCL Capstone Project

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    pydantic==2.7.1 \
    scikit-learn==1.5.1 \
    xgboost==2.0.3 \
    joblib==1.4.2 \
    numpy==1.26.4 \
    scipy==1.13.1

# Copy application files
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
