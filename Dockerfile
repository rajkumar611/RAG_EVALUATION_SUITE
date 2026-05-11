# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built deps from stage 1
COPY --from=deps /deps /usr/local

# Copy application source
COPY app.py main.py ./
COPY src/ src/
COPY frontend/ frontend/

# Pre-download the embedding model so first startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8080

# Disable reload in container; set ENV=production via docker run -e or compose
ENV ENV=production

CMD ["python", "main.py"]
