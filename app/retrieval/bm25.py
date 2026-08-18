"""Keyword retrieval with BM25 - the half of search that embeddings are bad at.

Dense vectors match meaning, which is exactly why they lose exact strings: a
question about `1GD-FTV` or `ISOFIX` gets embedded into "engine-ish" or
"child-seat-ish" space and the specific token stops being decisive. BM25 scores
the opposite way - rare terms that literally appear in a chunk dominate - so the
two retrievers fail on different questions. That is what makes fusing them worth
doing, rather than just tuning one of them harder.

Scope note for the interview: this builds the index in-process over all 1110
chunks (~2 MB of text, cheap). At millions of chunks the right answer is
Postgres full-text search (`tsvector` + GIN index) so the keyword side stays in
the database next to the vectors, instead of a Python object that every worker
has to rebuild on boot.
"""

import logging
import re

from rank_bm25 import BM25Okapi

from app.core.db import get_connection

logger = logging.getLogger(__name__)

_index: "BM25Index | None" = None       # module-level cache (lazy singleton)

# Split on anything that is not a letter or digit: keeps "1gd" and "ftv" as
# tokens, throws away the punctuation and the PDF's private-use icon glyphs.
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens.

    Deliberately no stemming and no stopword list: BM25 already discounts terms
    that appear in most documents (that is what IDF does), and stemming would
    damage the part numbers and model codes this retriever exists to catch.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """All chunk ids, texts and the fitted BM25 model, held together."""

    def __init__(self, ids: list[int], contents: list[str]) -> None:
        self.ids = ids
        self.contents = contents
        # BM25Okapi wants pre-tokenized documents and computes IDF at build time.
        self.model = BM25Okapi([tokenize(c) for c in contents])

    def search(self, query: str, k: int) -> list[tuple[int, float, str]]:
        """Return the top k (chunk_id, score, content), best first."""
        scores = self.model.get_scores(tokenize(query))
        # argsort without numpy: rank positions by score, take the best k.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], float(scores[i]), self.contents[i]) for i in order]


def get_index() -> BM25Index:
    """Build the index once per process, then reuse it.

    Rebuilding per request would re-read every chunk out of Postgres and refit
    IDF on every keystroke - the same lazy-singleton reason the embedding model
    is cached in app/ingestion/embed.py.
    """
    global _index
    if _index is None:
        with get_connection() as conn, conn.cursor() as cur:
            # Ordered by id so the index is deterministic across rebuilds, which
            # keeps eval runs comparable. Table-of-contents chunks are excluded:
            # every rare term in the manual also appears in its index, so BM25
            # matched leader lines as readily as real sections and that measurably
            # hurt hybrid recall until they were filtered out.
            cur.execute(
                """SELECT id, content FROM chunks
                   WHERE (metadata->>'noise') IS DISTINCT FROM 'true'
                   ORDER BY id;"""
            )
            rows = cur.fetchall()
        _index = BM25Index([r[0] for r in rows], [r[1] for r in rows])
        logger.info("bm25 index built over %d chunks", len(rows))
    return _index


def bm25_search(query: str, k: int = 5) -> list[dict]:
    """Keyword-only retrieval, in the same shape every retriever returns.

    Metadata is fetched for just the k winners rather than cached alongside the
    index: keeping one copy of the page numbers in Postgres avoids the two
    drifting apart after a re-ingest.
    """
    hits = get_index().search(query, k)
    if not hits:
        return []

    ids = [h[0] for h in hits]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, metadata FROM chunks WHERE id = ANY(%s);", (ids,))
        metadata = dict(cur.fetchall())

    return [
        {"id": cid, "content": content, "metadata": metadata[cid], "score": score}
        for cid, score, content in hits
    ]


if __name__ == "__main__":
    for hit in bm25_search("ISOFIX lower anchorages", k=5):
        print(f"{hit['score']:.2f}  p{hit['metadata']['page']:<5} "
              f"{hit['content'][:90].strip()}")
