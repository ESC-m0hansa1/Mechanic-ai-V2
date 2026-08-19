"""Application entrypoint: `uvicorn app.main:app`."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.db import get_connection
from app.core.observability import RequestIdFilter, log_requests
from app.core.warmup import warm_up

# Configure logging once, at startup, before anything logs. The request_id field
# comes from RequestIdFilter; it is "-" outside a request.
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s | %(message)s",
)
# Attach the filter to the ROOT handler so it applies to every logger in the
# process, ours and uvicorn's alike. Without it, %(request_id)s would raise.
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger(__name__)

# Built SPA (Component 9 / frontend). Absent in a bare checkout - the API must
# still start, so the mount below is conditional rather than assumed.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown. Everything before `yield` runs before serving traffic.

    lifespan replaces the deprecated @app.on_event("startup"): one context
    manager makes the paired setup/teardown obvious and runs teardown even when
    startup raised halfway through.
    """
    logger.info("starting %s strategy=%s", settings.app_name,
                settings.retrieval_strategy)
    warm_up()          # non-fatal by design; see app/core/warmup.py
    yield
    # Nothing to tear down: psycopg connections are per-request and the models
    # die with the process. Stated explicitly so the absence looks deliberate.
    logger.info("shutting down")


app = FastAPI(
    title=settings.app_name,
    description="RAG diagnostic assistant over a real vehicle service manual.",
    version="2.0.0",
    lifespan=lifespan,
)

# Order matters: middleware added last runs outermost, so this sees every
# request including CORS preflights and 422s from validation.
app.middleware("http")(log_requests)

# In production the SPA is served from this same origin, so CORS is not needed
# at all. It exists for `npm run dev` on :5173, which IS a different origin.
# Explicit list, not "*": a wildcard plus credentials is the classic mistake.
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

app.include_router(router)


@app.get("/api/health")
async def health():
    """Readiness: can we actually serve traffic (i.e. is the DB reachable)?

    Liveness and readiness are different questions - a container can be alive
    but useless. Load balancers and `docker compose` healthchecks want this one.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Two counts: total tells us ingestion ran, retrievable tells us
                # how many chunks retrieval can actually return after noise
                # flagging. A mismatch of ~8% is expected, not a fault.
                cur.execute("""
                    SELECT count(*),
                           count(*) FILTER (
                               WHERE (metadata->>'noise') IS DISTINCT FROM 'true'
                           )
                    FROM chunks;
                """)
                total, retrievable = cur.fetchone()
    except Exception as exc:
        logger.exception("health check failed")
        # 200 with status=degraded, not a 503: the endpoint's job is to REPORT
        # readiness. str(exc) is included because this is an operator-facing
        # endpoint on a private port, not a public one.
        return {"status": "degraded", "database": "unreachable", "error": str(exc)}

    return {
        "status": "ok",
        "database": "ok",
        "chunks_indexed": total,
        "chunks_retrievable": retrievable,
        "retrieval_strategy": settings.retrieval_strategy,
        "embedding_model": settings.embedding_model,
        "reranker_model": (
            settings.reranker_model
            if settings.retrieval_strategy == "reranked" else None
        ),
    }


# Mounted LAST so it cannot shadow /api/* or /docs. html=True makes unknown paths
# fall back to index.html, which is what a client-side router needs.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
    logger.info("serving SPA from %s", FRONTEND_DIST)
else:
    @app.get("/")
    async def root():
        """Liveness stand-in when no SPA has been built."""
        return {"app": settings.app_name, "status": "ok",
                "note": "frontend not built; API is at /api/ask and /docs"}
