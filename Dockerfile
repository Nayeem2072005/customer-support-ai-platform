# AI-Powered Customer Support Intelligence Platform - inference API
# Build:  docker build -t customer-support-api .
# Run:    docker run --rm -p 8000:8000 customer-support-api
FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by LightGBM and XGBoost on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/opt/nltk_data \
    PORT=8000

WORKDIR /app

# 1) Dependencies first (slow, rarely change -> cached between builds).
#    CPU-only PyTorch: the default Linux wheel bundles GPU libraries (GBs).
COPY requirements-api.txt .
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-api.txt

# 2) NLTK data used by src/preprocessing.py, downloaded at build time so the
#    container needs no internet at runtime.
RUN python -m nltk.downloader -d "$NLTK_DATA" stopwords wordnet punkt punkt_tab

# 3) Small model files and tokenizer/config from the repo ...
COPY models/ ./models/
# ... and the 268 MB DistilBERT weights from the Hugging Face Hub.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Nxm11/customer-support-distilbert', 'model.safetensors', local_dir='models/distilbert_ticket_type')"

# 4) Application code last (changes most often).
COPY src/ ./src/
COPY api/ ./api/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthy only when every model loaded (status == "healthy").
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, os, urllib.request as u; d = json.load(u.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')); raise SystemExit(0 if d['status'] == 'healthy' else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
