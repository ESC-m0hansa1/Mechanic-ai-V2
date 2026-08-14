import json
from collections import Counter
from psycopg.types.json import Json          # wraps a dict for a JSONB column
from app.core.db import get_connection
from app.ingestion.extract import extract_pages
from app.ingestion.chunk import chunk_page
from app.ingestion.embed import embed_texts

MIN_WORDS = 20        # drop fragments: a 1-word chunk is noise, not context


def log_distribution(chunks: list[dict]) -> None:
    counts = sorted(c["word_count"] for c in chunks)
    n = len(counts)
    print(f"chunks kept: {n} | min {counts[0]} | median {counts[n // 2]} | max {counts[-1]}")
    bins = Counter((c // 50) * 50 for c in counts)      # group into 50-word buckets
    for lo in sorted(bins):                             # crude ASCII histogram
        print(f"{lo:>4}-{lo + 49:<4} | {'#' * (bins[lo] * 40 // n)} {bins[lo]}")


def ingest(pdf_path: str, title: str) -> None:
    pages = extract_pages(pdf_path)
    # flatten pages -> chunks, keeping only chunks with enough words
    chunks = [c for i, p in enumerate(pages)
              for c in chunk_page(p, page=i + 1) if c["word_count"] >= MIN_WORDS]
    log_distribution(chunks)

    vectors = embed_texts([c["content"] for c in chunks])   # ONE batched call

    with get_connection() as conn:
        with conn.cursor() as cur:
            # RETURNING id gives us back the new row's id in the same round-trip
            cur.execute("INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id;",
                        (title, pdf_path))
            doc_id = cur.fetchone()[0]
            rows = [(doc_id, i, c["content"],
                     Json({"page": c["page"], "local_index": c["local_index"]}), v)
                    for i, (c, v) in enumerate(zip(chunks, vectors))]  # zip pairs them up
            cur.executemany(                                # one call, many inserts
                "INSERT INTO chunks (document_id, chunk_index, content, metadata, embedding)"
                " VALUES (%s, %s, %s, %s, %s);", rows)
    print(f"inserted {len(rows)} chunks as document id={doc_id}")


if __name__ == "__main__":
    ingest("data/manual.pdf", "Toyota HILUX Owner's Manual (India)")