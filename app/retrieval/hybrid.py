"""Hybrid retrieval: fuse the dense and BM25 rankings with Reciprocal Rank Fusion.

The problem RRF solves is that the two retrievers' scores are not comparable.
Cosine similarity lands in roughly 0.5-0.85 on this corpus; BM25 is an unbounded
sum of IDF terms that hit 21.65 on an `ISOFIX` query. Any weighted average of
the two numbers is really a weighted average of two different units, and the
weight would have to be re-tuned per corpus.

RRF throws the scores away and fuses *positions* instead:

    score(chunk) = sum over lists of  1 / (rrf_k + rank_in_that_list)

Being 1st in one list is worth 1/61 and 2nd is worth 1/62, so agreement between
the two retrievers matters far more than either one's confidence. rrf_k=60 is
the constant from the original TREC paper; it flattens the curve so the top few
ranks are not so dominant that a single retriever can force its favourite
through on its own.

Tradeoff, stated plainly: RRF cannot express "the dense hit was overwhelmingly
better than the keyword hit", because it never sees the margin. A tuned linear
combination of normalized scores can beat it - but only after you have enough
labelled data to tune it, and it has to be retuned when the corpus changes.
RRF needs no tuning, which is why it is the default here.
"""

import logging

from app.core.config import settings
from app.retrieval.bm25 import bm25_search
from app.retrieval.dense import dense_search

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[dict]], rrf_k: int, k: int
) -> list[dict]:
    """Fuse several ranked hit lists into one, best first.

    Pure function over hit lists: no DB, no model, so the ranking maths is
    unit-testable without any infrastructure.
    """
    fused: dict[int, float] = {}
    seen: dict[int, dict] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):     # ranks are 1-based
            fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank)
            # First list to produce a chunk supplies its content/metadata; both
            # lists carry identical text for the same id, so this is arbitrary.
            seen.setdefault(hit["id"], hit)

    best = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
    # The returned `score` is now an RRF score (~0.01-0.03), NOT a similarity.
    # Anything downstream that thresholds on score must be calibrated per
    # strategy - see eval/run_eval.py:calibrate_threshold.
    return [{**seen[cid], "score": fused[cid]} for cid in best]


def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """Dense + BM25, fused by RRF, top k returned."""
    # Each retriever nominates deeper than k so a chunk both retrievers rank
    # mid-list can beat one that only a single retriever loves. But not
    # arbitrarily deep: measured on the golden set, nominating 30 each was worse
    # than 10 each (MRR 0.691 vs 0.733), because RRF rewards co-occurrence and a
    # deep pool lets weak keyword matches accumulate credit. See fusion_depth.
    pool = max(settings.fusion_depth, k)
    dense_hits = dense_search(query, pool)
    keyword_hits = bm25_search(query, pool)
    return reciprocal_rank_fusion([dense_hits, keyword_hits], settings.rrf_k, k)


if __name__ == "__main__":
    for hit in hybrid_search("what fuel type does the diesel engine take?", k=5):
        print(f"{hit['score']:.4f}  p{hit['metadata']['page']:<5} "
              f"{hit['content'][:90].strip()}")
