"""The /api/ask contract, with retrieval and the LLM stubbed at the route seam.

What is being tested is the route's own behaviour: does it shape the response
correctly, does it refuse when there is nothing to ground an answer in, and does
it turn each kind of dependency failure into the right status code WITHOUT
leaking internals to the client. Those error paths are the ones that never get
exercised by hand.
"""

import pytest

from app.api import routes
from app.generation.llm import IDK, LLMError
from tests.conftest import make_hit

HITS = [make_hit(973, 693, 3.53, "If the engine will not start, check the battery."),
        make_hit(975, 694, 1.20, "The starter motor turns over slowly.")]


@pytest.fixture
def stub_deps(monkeypatch):
    """Replace the two functions the route calls. Nothing below them runs."""
    def install(hits=HITS, text="Check the battery (p. 693)."):
        monkeypatch.setattr(routes, "search", lambda q, k: hits)
        monkeypatch.setattr(routes, "answer", lambda q, chunks: text)
    return install


def test_happy_path_returns_answer_and_citable_sources(client, stub_deps):
    stub_deps()
    resp = client.post("/api/ask", json={"question": "engine will not start"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Check the battery (p. 693)."
    assert body["refused"] is False
    # Sources exist so the answer is auditable - a page number the user can open.
    assert [s["page"] for s in body["sources"]] == [693, 694]
    assert body["sources"][0]["chunk_id"] == 973
    assert body["strategy"]                      # which strategy served this
    assert body["retrieval_ms"] <= body["total_ms"]


def test_refusal_is_flagged_for_the_frontend(client, stub_deps):
    stub_deps(text=IDK)
    body = client.post("/api/ask", json={"question": "how do I bake bread"}).json()
    assert body["refused"] is True


def test_empty_retrieval_never_reaches_the_llm(client, monkeypatch):
    # The real answer() is left in place: with no chunks it must short-circuit to
    # IDK rather than pay for a call that could only hallucinate. If it tried to
    # reach the network this test would fail on a connection error.
    monkeypatch.setattr(routes, "search", lambda q, k: [])
    body = client.post("/api/ask", json={"question": "capital of France"}).json()
    assert body["answer"] == IDK
    assert body["refused"] is True
    assert body["sources"] == []


def test_llm_failure_is_a_502_with_no_internal_detail(client, monkeypatch):
    monkeypatch.setattr(routes, "search", lambda q, k: HITS)
    def boom(q, chunks):
        raise LLMError("provider returned 429 for key sk-abc123")
    monkeypatch.setattr(routes, "answer", boom)

    resp = client.post("/api/ask", json={"question": "engine will not start"})

    assert resp.status_code == 502
    assert resp.json()["detail"] == "language model unavailable"
    assert "sk-abc123" not in resp.text        # provider detail stays in the logs


def test_retrieval_failure_is_a_503_with_no_internal_detail(client, monkeypatch):
    def boom(q, k):
        raise RuntimeError('connection to server at "db" failed: password auth')
    monkeypatch.setattr(routes, "search", boom)

    resp = client.post("/api/ask", json={"question": "engine will not start"})

    assert resp.status_code == 503
    assert resp.json()["detail"] == "retrieval unavailable"
    assert "password" not in resp.text


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                        # under min_length=3
    {"question": "x" * 501},                   # over max_length=500
    {"question": "valid question", "k": 0},    # k below ge=1
    {"question": "valid question", "k": 99},   # k above le=20
    {},                                        # question missing entirely
])
def test_bad_input_is_rejected_before_any_work_happens(client, monkeypatch, payload):
    # Pydantic validation runs before the handler, so retrieval must never be
    # called. Guarding an expensive pipeline with a 422 is the whole point.
    monkeypatch.setattr(routes, "search",
                        lambda q, k: pytest.fail("retrieval ran on invalid input"))
    assert client.post("/api/ask", json=payload).status_code == 422


def test_request_id_is_echoed_for_correlation(client, stub_deps):
    stub_deps()
    resp = client.post("/api/ask", json={"question": "engine will not start"},
                       headers={"X-Request-ID": "trace-me-42"})
    # A client-supplied id survives into our logs and back out, so a bug report
    # can quote one string that appears on every line of the request.
    assert resp.headers["X-Request-ID"] == "trace-me-42"


def test_request_id_is_generated_when_absent(client, stub_deps):
    stub_deps()
    resp = client.post("/api/ask", json={"question": "engine will not start"})
    assert resp.headers["X-Request-ID"] not in ("", "-")
