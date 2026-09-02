# syntax=docker/dockerfile:1
#
# Three stages, because neither the frontend's toolchain nor the model exporter's
# has any business in the runtime image.
#
#   frontend  Node builds the SPA to static files
#   exporter  PyTorch converts both checkpoints to ONNX graphs
#   runtime   Python serves the static files and the ONNX graphs
#
# Nothing Node-shaped and nothing torch-shaped survives into production, which is
# why "one container behind one public URL" still holds even though the UI is
# React - and why the image fits a 512 MB service (see stage 3).

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


# ---------- stage 2: export the models to ONNX ----------
# This stage exists purely so torch never reaches the runtime image. It is also
# where the CUDA trap lives: a plain `pip install torch` on Linux resolves to the
# GPU build and drags in ~2.5 GB of NVIDIA libraries that a CPU host can never
# use, so the wheel comes from PyTorch's CPU index explicitly.
FROM python:3.12-slim AS exporter

# HF_HOME is the checkpoint download cache; it is discarded with this stage.
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /build
# CPU-only torch FIRST, from PyTorch's own index. This ordering is the single
# most important line in the stage: `pip install torch` on Linux resolves to the
# CUDA build and drags in ~2.5 GB of NVIDIA libraries a CPU host can never use.
# --index-url (not --extra-index-url) so PyPI is not consulted for torch at all;
# with both indexes live pip picks by version and would take the CUDA wheel.
RUN pip install --no-cache-dir torch==2.13.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-export.txt ./
RUN pip install --no-cache-dir -r requirements-export.txt

COPY scripts/export_onnx.py ./scripts/
# Downloads both checkpoints and traces them to ONNX. Weights are fp32, matching
# torch's arithmetic, because the retrieval thresholds in eval/ are calibrated
# against torch scores and int8 quantization would move them.
RUN python scripts/export_onnx.py --out /build/models/onnx


# ---------- stage 3: runtime ----------
FROM python:3.12-slim AS runtime

# PYTHONDONTWRITEBYTECODE: no .pyc litter in a layer that is read-only anyway.
# PYTHONUNBUFFERED: logs reach docker logs immediately instead of sitting in a
# buffer, which is the difference between debuggable and not.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
# The target platform allocates a fraction of a CPU. Letting ONNX Runtime spawn
# one thread per visible core there produces threads contending for a slice none
# of them can fill, which is slower than staying single-threaded. Override to 0
# (ORT's default) when running on a real box.
ENV ONNX_INTRA_THREADS=1

WORKDIR /srv

# No torch here - that is the entire point of stage 2. Measured resident memory
# of this server on the old sentence-transformers stack was 509 MB against a
# 512 MB cap, and only ~139 MB of it was model weights; the rest was torch and
# transformers being imported. See app/core/onnx_backend.py.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY eval/ ./eval/
COPY data/ ./data/
# The ONNX graphs and their tokenizers (~223 MB), baked in rather than downloaded
# on first boot: baking makes startup independent of Hugging Face being
# reachable, and a cold start does not re-fetch them on every new container.
COPY --from=exporter /build/models/onnx ./models/onnx
# The built SPA lands where app/main.py looks for it (FRONTEND_DIST).
COPY --from=frontend /ui/dist ./frontend/dist

# Run as a non-root user: a container escape should not land on root, and nothing
# here needs write access to the filesystem.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Hits the readiness endpoint, which checks Postgres too - so an app that is up
# but cannot reach its database is correctly reported as unhealthy.
# start-period covers the warm-up (model load + BM25 index build).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,sys,urllib.request,json; \
r=json.load(urllib.request.urlopen('http://localhost:%s/api/health' % os.getenv('PORT','8000'))); \
sys.exit(0 if r.get('status')=='ok' else 1)"

# No --reload, no --workers. Reload is a dev feature. Workers stay at 1 because
# each one would load its own copy of both models; horizontal scaling belongs to
# the orchestrator, which can at least put them on different machines.
#
# Shell form, so $PORT is expanded at run time: hosted platforms assign the port
# and inject it, and a hardcoded 8000 would make the service unreachable there.
# The default keeps `docker compose up` and local runs working unchanged.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
