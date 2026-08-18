"""Backfill: mark existing chunks that are table-of-contents noise.

Why a flag rather than a DELETE and re-ingest: chunk ids are the primary key the
golden set is labelled against. Re-ingesting renumbers every chunk and silently
invalidates 24 hand-labelled questions, which is a far worse outcome than
carrying 89 dead rows. The flag also keeps provenance - the text is still there
to audit if the heuristic turns out to be wrong.

Idempotent: running it twice sets the same flags. New ingests never need it
because app/ingestion/store.py applies the same filter at insert time.

    python -m app.ingestion.flag_noise
"""

import logging

from app.core.db import get_connection
from app.ingestion.quality import is_index_noise

logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, content FROM chunks ORDER BY id;")
        rows = cur.fetchall()

        noisy = [(cid,) for cid, content in rows if is_index_noise(content)]
        # jsonb || jsonb merges keys, so this adds "noise": true without
        # clobbering the page/chunk metadata already stored.
        cur.executemany(
            """UPDATE chunks
               SET metadata = metadata || '{"noise": true}'::jsonb
               WHERE id = %s;""",
            noisy,
        )
        # Anything previously flagged that no longer matches gets un-flagged, so
        # re-running after a threshold change converges instead of accumulating.
        cur.execute(
            """UPDATE chunks
               SET metadata = metadata - 'noise'
               WHERE metadata ? 'noise' AND NOT (id = ANY(%s));""",
            ([cid for (cid,) in noisy],),
        )
        conn.commit()

    logger.info("flagged %d of %d chunks as index noise (%.1f%%)",
                len(noisy), len(rows), 100 * len(noisy) / max(len(rows), 1))


if __name__ == "__main__":
    main()
