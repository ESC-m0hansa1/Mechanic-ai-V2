"""rerank() is pure over hit lists, so it is tested without loading the model.

The stub returns scores I choose, which is the point: these tests check the
plumbing (ordering, truncation, shape preservation), not whether MiniLM is any
good. That question is answered by eval/, against labelled data.
"""

import pytest

from app.retrieval import rerank as rerank_mod
from tests.conftest import make_hit


class StubEncoder:
    """Stands in for CrossEncoder; returns a fixed score per pair."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return self.scores[: len(pairs)]


@pytest.fixture
def stub(monkeypatch):
    def install(scores):
        encoder = StubEncoder(scores)
        monkeypatch.setattr(rerank_mod, "get_reranker", lambda: encoder)
        return encoder
    return install


def test_reranking_overrides_the_input_order(stub):
    # Stage 1 handed these over in the order 1, 2, 3. The cross-encoder says the
    # third one is the best. That reversal is the entire value of this stage.
    stub([-4.0, 1.0, 9.0])
    candidates = [make_hit(1, 10, 0.9), make_hit(2, 20, 0.8), make_hit(3, 30, 0.7)]

    out = rerank_mod.rerank("q", candidates, k=3)

    assert [h["id"] for h in out] == [3, 2, 1]
    assert out[0]["score"] == 9.0        # score is replaced by the logit


def test_result_is_truncated_to_k(stub):
    stub([1.0, 2.0, 3.0, 4.0])
    out = rerank_mod.rerank("q", [make_hit(i, i, 0.5) for i in range(1, 5)], k=2)
    assert [h["id"] for h in out] == [4, 3]


def test_query_is_paired_with_every_candidate(stub):
    encoder = stub([1.0, 2.0])
    rerank_mod.rerank("how do I check the oil?",
                      [make_hit(1, 10, 0.9, "oil text"),
                       make_hit(2, 20, 0.8, "tyre text")], k=2)
    # A cross-encoder scores (query, passage) PAIRS - passing chunks alone would
    # silently produce meaningless scores rather than an error.
    assert encoder.calls[0] == [("how do I check the oil?", "oil text"),
                                ("how do I check the oil?", "tyre text")]


def test_content_and_page_survive_reranking(stub):
    stub([5.0])
    out = rerank_mod.rerank("q", [make_hit(42, 567, 0.81, "Check the oil.")], k=1)
    assert out[0]["metadata"]["page"] == 567
    assert out[0]["content"] == "Check the oil."


def test_empty_candidates_short_circuits_without_loading_the_model(monkeypatch):
    # No stub installed: if rerank() touched get_reranker() here it would try to
    # load a 90 MB model for zero pairs.
    monkeypatch.setattr(rerank_mod, "get_reranker",
                        lambda: pytest.fail("model loaded for an empty pool"))
    assert rerank_mod.rerank("q", [], k=5) == []
