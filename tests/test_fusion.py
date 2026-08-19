"""Reciprocal Rank Fusion is the piece of arithmetic the hybrid strategy rests on.

It is also pure - list of hits in, list of hits out, no DB and no model - which
is exactly why it was written as a standalone function. These tests pin the
properties the docstring in app/retrieval/hybrid.py claims, so a "harmless"
refactor that starts fusing scores again fails here instead of quietly costing
two points of MRR.
"""

from app.retrieval.hybrid import reciprocal_rank_fusion
from tests.conftest import make_hit

RRF_K = 60


def test_agreement_between_retrievers_beats_a_single_confident_vote():
    # THE claim of RRF. Chunk 2 is only mid-list in both lists; chunk 1 is first
    # in one list and absent from the other. Two mid votes must win.
    dense = [make_hit(1, 10, 0.9), make_hit(9, 90, 0.8), make_hit(2, 20, 0.7)]
    keyword = [make_hit(8, 80, 15.0), make_hit(7, 70, 12.0), make_hit(2, 20, 9.0)]

    fused = reciprocal_rank_fusion([dense, keyword], RRF_K, k=3)

    assert fused[0]["id"] == 2               # 1/63 + 1/63 == 0.0317
    assert fused[0]["score"] > fused[1]["score"]


def test_only_positions_matter_not_score_magnitudes():
    # Cosine similarities live in 0.5-0.85, BM25 reached 21.65 on this corpus.
    # Fusing the two numbers directly would be a weighted average of different
    # units, so RRF must produce the same answer whatever the scores say.
    order_a = [make_hit(1, 10, 0.51), make_hit(2, 20, 0.50)]
    order_b = [make_hit(1, 10, 21.65), make_hit(2, 20, 0.01)]

    assert ([h["id"] for h in reciprocal_rank_fusion([order_a], RRF_K, 2)]
            == [h["id"] for h in reciprocal_rank_fusion([order_b], RRF_K, 2)])


def test_returned_score_is_the_rrf_sum_not_the_input_similarity():
    # Documented contract: downstream code must NOT treat this as a similarity.
    # A single list, rank 1 -> 1/(60+1).
    fused = reciprocal_rank_fusion([[make_hit(1, 10, 0.83)]], RRF_K, k=1)
    assert fused[0]["score"] == 1.0 / 61
    assert fused[0]["score"] != 0.83


def test_ranks_are_one_based():
    # If the loop used enumerate() without start=1, the top hit would score
    # 1/(60+0) instead of 1/61 - a silent off-by-one that inflates rank 1.
    fused = reciprocal_rank_fusion([[make_hit(1, 10, 0.9)]], rrf_k=0, k=1)
    assert fused[0]["score"] == 1.0          # 1/(0+1), not a ZeroDivisionError


def test_larger_rrf_k_flattens_the_advantage_of_being_first():
    def gap(rrf_k: int) -> float:
        hits = reciprocal_rank_fusion(
            [[make_hit(1, 10, 0.9), make_hit(2, 20, 0.8)]], rrf_k, k=2
        )
        return hits[0]["score"] - hits[1]["score"]

    # This is what rrf_k=60 buys: one retriever cannot force its favourite
    # through on its own, because rank 1 is worth barely more than rank 2.
    assert gap(60) < gap(5) < gap(0)


def test_result_is_truncated_to_k():
    lists = [[make_hit(i, i, 1.0 / i) for i in range(1, 11)]]
    assert len(reciprocal_rank_fusion(lists, RRF_K, k=3)) == 3


def test_empty_input_yields_empty_output():
    # Both retrievers can legitimately return nothing (empty corpus, or every
    # candidate filtered as noise). The route must get [] and short-circuit to
    # "I don't know", not crash.
    assert reciprocal_rank_fusion([], RRF_K, k=5) == []
    assert reciprocal_rank_fusion([[], []], RRF_K, k=5) == []


def test_content_and_metadata_survive_fusion():
    # The frontend cites metadata["page"]; losing it here would break citations
    # while every score still looked plausible.
    dense = [make_hit(42, 567, 0.81, content="Check the engine oil.")]
    fused = reciprocal_rank_fusion([dense], RRF_K, k=1)
    assert fused[0]["metadata"]["page"] == 567
    assert fused[0]["content"] == "Check the engine oil."
