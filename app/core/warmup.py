"""Startup warm-up: pay the model-loading cost before the first user does.

Every expensive object in this app is a lazy singleton - the embedding model,
the BM25 index, the cross-encoder. That is the right default for scripts and
tests, but in a server it means the FIRST request after boot pays for all of it:
measured at 6.4 s against a warm steady state of ~600 ms. The user who happens
to arrive first gets a request ten times slower than everyone else, and a
container orchestrator that health-checks with a timeout may kill the pod before
it ever answers.

So we do the same throwaway query the eval harness does, once, at startup.

Deliberately non-fatal. If the database is not up yet - normal during
`docker compose up`, where the app can win the race against Postgres - a crash
here would take the whole process down and a restart loop would look like a bug
in the app. Instead we log it and let /health report the truth.
"""

import logging
import time

logger = logging.getLogger(__name__)


def warm_up() -> float | None:
    """Run one throwaway query through the configured strategy.

    Returns the elapsed seconds, or None if warm-up failed (which is survivable).
    Imports live inside the function so that importing this module - as the tests
    do - never drags in torch.
    """
    from app.core.config import settings
    from app.retrieval.search import search

    t0 = time.perf_counter()
    try:
        # Not a real question: the point is to force the model loads and the
        # first DB round-trip, and to exercise whichever strategy is configured
        # rather than a hard-coded one.
        search("warm up the retrieval stack", 1)
    except Exception:
        logger.warning(
            "warm-up failed for strategy=%s; first real request will be slow "
            "and /health will report the cause",
            settings.retrieval_strategy,
            exc_info=True,
        )
        return None

    elapsed = time.perf_counter() - t0
    logger.info("warm-up complete strategy=%s in %.2fs",
                settings.retrieval_strategy, elapsed)
    return elapsed
