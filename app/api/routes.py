"""The /api/ask endpoint: retrieve -> prompt -> generate -> return answer + sources."""

import logging
import time

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.api.schemas import AskRequest, AskResponse, SourceChunk
from app.core.config import settings
from app.generation.llm import IDK, LLMError, answer
from app.retrieval.search import search

logger = logging.getLogger(__name__)
# Every endpoint lives under /api so the built SPA can own "/" without the two
# route tables ever competing. Changing this one string moves the whole API.
router = APIRouter(prefix="/api", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Answer a repair question from the ingested manual.

    The route is async so the server can handle other requests while this one
    waits. Retrieval and the LLM call are *blocking* (psycopg, torch, httpx),
    so we hand them to a worker thread with run_in_threadpool - calling them
    directly would freeze the whole event loop for every other user.
    """
    t0 = time.perf_counter()
    try:
        hits = await run_in_threadpool(search, req.question, req.k)
    except Exception:
        # exc_info=True logs the full traceback server-side...
        logger.exception("retrieval failed", extra={"question": req.question})
        # ...while the client gets a generic message (never leak internals).
        raise HTTPException(status_code=503, detail="retrieval unavailable")

    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    try:
        text = await run_in_threadpool(answer, req.question, hits)
    except LLMError as exc:
        logger.warning("llm call failed: %s", exc)
        raise HTTPException(status_code=502, detail="language model unavailable")

    total_ms = int((time.perf_counter() - t0) * 1000)
    # IDK is a module constant in llm.py precisely so this comparison is not a
    # magic string duplicated in two places.
    refused = text.strip() == IDK
    logger.info(
        "answered question=%r hits=%d refused=%s retrieval_ms=%d total_ms=%d",
        req.question, len(hits), refused, retrieval_ms, total_ms,
    )

    return AskResponse(
        question=req.question,
        answer=text,
        refused=refused,
        sources=[
            SourceChunk(
                chunk_id=h["id"],
                page=h["metadata"]["page"],
                score=round(float(h["score"]), 4),
                preview=h["content"][:300],
            )
            for h in hits
        ],
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
        strategy=settings.retrieval_strategy,
    )
