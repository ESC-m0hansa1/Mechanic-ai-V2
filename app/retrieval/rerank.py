"""Cross-encoder reranking: a second, expensive pass over a small candidate pool.

The retrievers before this one are *bi-encoders*: the query and the chunk are
embedded separately and compared with a dot product. That is what makes them
fast - every chunk vector was computed once at ingest time and never depends on
the question - and it is also their ceiling. The two texts never meet inside the
model, so the encoder has to compress a whole page of the manual into 384
numbers without knowing what will be asked of it.

A cross-encoder concatenates them - `[CLS] query [SEP] chunk [SEP]` - and runs
the pair through the transformer together, with attention connecting every query
token to every chunk token. It outputs one relevance logit. That is strictly
more informative and strictly more expensive: nothing can be precomputed, so
cost is one forward pass per (query, chunk) pair. Measured on this box (CPU
only, chunks averaging 820 characters): ~72 ms per pair, so scoring all 1021
retrievable chunks would take over a minute. Eight is ~580 ms.

Hence the standard two-stage shape, and the reason `hybrid` exists even though
it lost to `dense` on aggregate MRR (0.733 vs 0.757):

    stage 1  hybrid_search  -> candidate_pool hits  cheap, tuned for RECALL
    stage 2  cross-encoder  -> reorder, keep k      expensive, tuned for PRECISION

Stage 1 only has to get the right chunk *somewhere* in the pool. Stage 2 decides
the order. Splitting the objectives is what lets stage 1 use BM25 at all: BM25's
lexical noise ruins ordering (paraphrase MRR 0.528 -> 0.444) but it finds exact
strings like `ISOFIX` and `1GD-FTV` that the embedding model smooths away. Give
those hits to the cross-encoder and its noise costs less, because a chunk that
merely shares rare words with the question scores low once the model reads both
together.

Two costs, stated honestly:

* Latency. p50 goes 108 ms (dense) -> 576 ms, and a second ~90 MB model is
  pinned in memory. Acceptable here only because the LLM call after it takes
  seconds anyway; on a retrieval-only API it would not be.
* The pool is a real tradeoff, not a free dial. Handing the reranker MORE
  candidates measured WORSE on this corpus - see config.candidate_pool.

Worth it only if the measured ranking gain is real - see eval/results/reranked.json.
"""

import logging

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None       # lazy singleton, same reason as embed.get_model


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once and reuse it.

    ms-marco-MiniLM-L-6-v2 is a 6-layer, ~90 MB model trained on MS MARCO
    passage ranking. It is the small end of the family on purpose: L-12 and the
    monoT5 rerankers score better on BEIR but cost 2-10x the latency, and this
    is a synchronous step inside a request, not a batch job.
    """
    global _model
    if _model is None:
        # max_length=512 matches the chunker's window: a longer limit would just
        # pad, a shorter one would silently truncate the end of every chunk.
        _model = CrossEncoder(settings.reranker_model, max_length=512)
    return _model


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Reorder `candidates` by cross-encoder relevance and return the best k.

    Pure function over hit lists (no DB), so it can be tested with fabricated
    candidates the same way reciprocal_rank_fusion is.
    """
    if not candidates:
        return []

    pairs = [(query, hit["content"]) for hit in candidates]
    # One batched forward pass, not a loop: batching is where the SIMD/GPU win
    # is, and the whole pool fits in a single batch at this size.
    scores = get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)

    # `score` is now a raw logit, typically about -11 (irrelevant) to +11
    # (clearly relevant) - NOT a probability and NOT comparable to the cosine
    # similarities the dense strategy returns. Any threshold on it has to be
    # calibrated for this strategy (eval/run_eval.py:calibrate_threshold).
    reranked = [{**hit, "score": float(s)} for hit, s in zip(candidates, scores)]
    reranked.sort(key=lambda h: h["score"], reverse=True)
    return reranked[:k]


def reranked_search(query: str, k: int = 5) -> list[dict]:
    """Hybrid for recall, cross-encoder for precision."""
    # Ask stage 1 for candidate_pool, not k: the point is to hand the reranker a
    # few chunks hybrid ranked just below the cut, where answers RRF mis-ordered
    # are sitting. Never fewer than k, or we could not fill the page.
    pool = max(settings.candidate_pool, k)
    candidates = hybrid_search(query, pool)
    logger.debug("rerank: %d candidates -> top %d", len(candidates), k)
    return rerank(query, candidates, k)


if __name__ == "__main__":
    # The query that motivated this stage: pure paraphrase, no manual wording.
    for hit in reranked_search("my truck won't turn over", k=5):
        print(f"{hit['score']:+7.3f}  p{hit['metadata']['page']:<5} "
              f"{hit['content'][:88].strip()}")
