import numpy as np
from app.core.db import get_connection
from app.ingestion.embed import embed_texts

# bge-small is trained with this instruction on QUERIES only, not on stored passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def dense_search(query: str, k: int = 5) -> list[dict]:
    """Return the k chunks whose embeddings are nearest to the query."""
    # Embed the query with the SAME model as the chunks (else vectors aren't comparable).
    qvec = embed_texts([QUERY_PREFIX + query])[0]
    # float32 numpy array: pgvector's register_vector() adapts this to a `vector` param.
    qvec = np.array(qvec, dtype=np.float32)

    # `<=>` is pgvector's COSINE DISTANCE operator (0 = identical, 2 = opposite).
    # Since our vectors are normalized, similarity = 1 - distance.
    # The WHERE clause drops table-of-contents chunks (app/ingestion/quality.py).
    # IS DISTINCT FROM handles NULL: a chunk with no "noise" key must still match.
    sql = """
        SELECT id, content, metadata, 1 - (embedding <=> %s) AS score
        FROM chunks
        WHERE (metadata->>'noise') IS DISTINCT FROM 'true'
        ORDER BY embedding <=> %s   -- DB does the nearest-neighbour sort, not Python
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (qvec, qvec, k))   # params passed separately = no SQL injection
            rows = cur.fetchall()

    # Turn raw tuples into readable dicts (r[0]=id, r[1]=content, r[2]=metadata, r[3]=score)
    return [{"id": r[0], "content": r[1], "metadata": r[2], "score": float(r[3])}
            for r in rows]


if __name__ == "__main__":
    for hit in dense_search("how do I check the engine oil level?", k=3):
        print(f"[{hit['score']:.3f}] page {hit['metadata']['page']}: {hit['content'][:110]}...")