# syntax=docker/dockerfile:1
#
# Two stages, because the frontend's toolchain has no business in the runtime
# image. Node builds the SPA to static files; the Python image copies those files
# and serves them. Nothing Node-shaped survives into production - no node_modules,
# no Node process - which is why "one container behind one public URL" still holds
# even though the UI is React.

# ---------- stage 1: build the SPA ----------
FROM node:22-alpine AS frontend

WORKDIR /ui
# Copy manifests first so `npm ci` is cached and only re-runs when dependencies
# actually change, not on every source edit.
COPY frontend/package.json frontend/package-lock.json ./
# `ci` not `install`: it installs exactly the lockfile and fails if the manifest
# and lockfile disagree, which is what reproducible means.
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build            # -> /ui/dist


# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE: no .pyc litter in a layer that is read-only anyway.
# PYTHONUNBUFFERED: logs reach docker logs immediately instead of sitting in a
# buffer, which is the difference between debuggable and not.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /srv

# CPU-only torch, installed BEFORE requirements.txt and from PyTorch's own index.
# This is the single biggest decision in this file: `pip install
# sentence-transformers` on Linux resolves torch to the CUDA build and drags in
# ~2.5 GB of NVIDIA libraries that can never be used on a CPU host. Pinning the
# +cpu wheel first means the later resolve sees torch as already satisfied.
RUN pip install --no-cache-dir torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image (~220 MB). The alternative is downloading
# them on first boot, which makes startup depend on Hugging Face being reachable,
# adds a minute to a cold start, and re-downloads on every new container. Baking
# them makes the image bigger and the runtime deterministic; for a service that
# scales to a handful of instances that is the right side of the trade.
RUN python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY

COPY app/ ./app/
COPY eval/ ./eval/
COPY data/ ./data/
# The built SPA lands where app/main.py looks for it (FRONTEND_DIST).
COPY --from=frontend /ui/dist ./frontend/dist

# Run as a non-root user: a container escape should not land on root, and nothing
# here needs write access to the filesystem.
RUN useradd --create-home --uid 10001 appuser \
    && chmod -R a+rX /opt/models \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Hits the readiness endpoint, which checks Postgres too - so an app that is up
# but cannot reach its database is correctly reported as unhealthy.
# start-period covers the warm-up (model load + BM25 index build).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys,urllib.request,json; \
r=json.load(urllib.request.urlopen('http://localhost:8000/api/health')); \
sys.exit(0 if r.get('status')=='ok' else 1)"

# No --reload, no --workers. Reload is a dev feature. Workers stay at 1 because
# each one would load its own copy of both models; horizontal scaling belongs to
# the orchestrator, which can at least put them on different machines.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
