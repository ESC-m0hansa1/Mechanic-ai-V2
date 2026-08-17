"""Dev tool for building the golden set BY HAND, not by trusting the retriever.

Labelling relevance from whatever dense search already returns would be
circular: the eval would then measure "does dense search agree with dense
search". So this exposes two independent discovery paths and I read the text
before writing any chunk id into golden_set.json:

    python -m eval.inspect_chunks grep "engine oil"      # SQL keyword scan
    python -m eval.inspect_chunks ask "how do I check the oil?"   # dense search
    python -m eval.inspect_chunks show 805 806 807       # full text of chunks

Usage is documented here because "how did you build your eval set" is a
question every interviewer asks about a metrics table.
"""

import sys

from app.core.db import get_connection


def grep(term: str, limit: int = 15) -> None:
    """Keyword scan straight over the stored text - no embeddings involved."""
    sql = """
        SELECT id, metadata->>'page', left(content, 160)
        FROM chunks
        WHERE content ILIKE %s
        ORDER BY id
        LIMIT %s;
    """
    with get_connection() as conn, conn.cursor() as cur:
        # %term% -> substring match; parameterised so the term can't inject SQL
        cur.execute(sql, (f"%{term}%", limit))
        for cid, page, preview in cur.fetchall():
            print(f"id={cid:<5} p.{page:<4} {preview.strip()[:150]}")


def show(ids: list[int]) -> None:
    """Print chunks in full so I can judge relevance on the actual wording."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, metadata->>'page', content FROM chunks WHERE id = ANY(%s);",
            (ids,),
        )
        for cid, page, content in cur.fetchall():
            print(f"\n{'=' * 70}\nid={cid}  page={page}\n{'=' * 70}\n{content}")


def ask(question: str, k: int = 8) -> None:
    """What the current strategy returns - a second, independent candidate list."""
    from app.retrieval.search import search

    for hit in search(question, k):
        page = hit["metadata"]["page"]
        print(f"id={hit['id']:<5} p.{page:<4} {hit['score']:.3f}  "
              f"{hit['content'][:120].strip()}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["grep", "oil"]
    if cmd == "grep":
        grep(" ".join(rest))
    elif cmd == "show":
        show([int(x) for x in rest])
    elif cmd == "ask":
        ask(" ".join(rest))
    else:
        print(__doc__)
