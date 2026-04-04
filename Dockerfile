# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System dependencies: Tesseract OCR + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download NLP models ───────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm && \
    python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('punkt_tab', quiet=True);"

# Pre-download sentence-transformer model to bake it into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# Create runtime directories
RUN mkdir -p data/raw_resumes models database

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8501   
# Streamlit

EXPOSE 8000   
# FastAPI (optional)

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
