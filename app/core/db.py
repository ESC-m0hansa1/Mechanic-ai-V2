"""Postgres access: one pooled connection source for the whole process.

Originally this opened a fresh connection per call. Against local Postgres that
costs ~1 ms and the design flaw is invisible. Against a managed database in
another region it is 2-3 s, because TLS plus SCRAM authentication takes several
round trips and `reranked` retrieval makes two or three calls per request. So
the connections are pooled and the handshake is paid once, at startup, instead
of on every question.

The pool is created lazily for the same reason the embedding model is: importing
this module must not require a reachable database, or the test suite and the
ingestion CLI would both need one.
"""

import atexit
import logging

import psycopg
from pgvector.psycopg import register_vector  # adapter for the vector(384) type
from psycopg_pool import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Run once per newly-opened connection, not once per checkout."""
    register_vector(conn)  # so we can send/receive `vector` values as numpy arrays


def get_pool() -> ConnectionPool:
    """Build the pool once per process, then reuse it."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,      # one warm connection is enough for a single worker
            max_size=6,      # ceiling: retrieval is 2-3 short queries per request
            configure=_configure,
            # Managed databases suspend idle compute (Neon does it after 5 min)
            # and drop the socket. Without this check the pool would hand out a
            # dead connection and the first question after a quiet spell would
            # fail; check_connection discards it and opens a fresh one instead.
            check=ConnectionPool.check_connection,
            max_idle=240.0,     # under Neon's 5-minute suspend window
            max_lifetime=3600.0,
            timeout=30.0,       # how long a caller waits for a free connection
            open=True,
        )
        # Short-lived CLI entry points (ingestion, the __main__ blocks below)
        # would otherwise leave the pool's worker threads running at exit.
        atexit.register(_close_pool)
        logger.info("db pool opened (min=1 max=6)")
    return _pool


def _close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_connection():
    """Borrow a connection from the pool.

    Returns a context manager, so every existing `with get_connection() as conn:`
    call site keeps working unchanged - the difference is that leaving the block
    returns the connection to the pool instead of closing it. Commit-on-success
    and rollback-on-exception behave as before.
    """
    return get_pool().connection()


if __name__ == "__main__":
    with get_connection() as conn:
        with conn.cursor() as cur:          # a cursor executes SQL and holds results
            cur.execute("SELECT count(*) FROM chunks;")
            print("chunks in DB:", cur.fetchone()[0])  # fetchone() -> one result row
