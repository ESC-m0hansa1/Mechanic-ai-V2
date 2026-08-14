"""Application entrypoint: `uvicorn app.main:app`."""

import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.db import get_connection

# Configure logging once, at startup, before anything logs.
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    description="RAG diagnostic assistant over a real vehicle service manual.",
    version="2.0.0",
)
app.include_router(router)


@app.get("/")
async def root():
    """Liveness: is the process up?"""
    return {"app": settings.app_name, "status": "ok"}


@app.get("/health")
async def health():
    """Readiness: can we actually serve traffic (i.e. is the DB reachable)?

    Liveness and readiness are different questions - a container can be alive
    but useless. Load balancers and `docker compose` healthchecks want this one.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks;")
                chunk_count = cur.fetchone()[0]
    except Exception as exc:
        return {"status": "degraded", "database": "unreachable", "error": str(exc)}

    return {
        "status": "ok",
        "database": "ok",
        "chunks_indexed": chunk_count,
        "retrieval_strategy": settings.retrieval_strategy,
        "embedding_model": settings.embedding_model,
    }
