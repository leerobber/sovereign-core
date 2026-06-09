# ╔══════════════════════════════════════════════╗
# ║  Sovereign Core — Production Dockerfile      ║
# ║  Multi-stage: deps → test → production       ║
# ╚══════════════════════════════════════════════╝

FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && apt-get clean

# ── Dependencies stage ──────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Test stage ──────────────────────────────────
FROM deps AS test
COPY . .
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-mock httpx
RUN python -m pytest tests/ -v --tb=short

# ── Production stage ────────────────────────────
FROM deps AS production

COPY . .

# Non-root user
RUN useradd -m -u 1000 sovereign && chown -R sovereign:sovereign /app
USER sovereign

RUN mkdir -p /app/data/gateway /app/data/patterns /app/data/ledger

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "gateway.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--access-log"]
