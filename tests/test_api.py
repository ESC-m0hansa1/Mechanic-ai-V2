"""Route-level tests: the contract the frontend depends on, and the failure modes.

Retrieval and the LLM are stubbed at `app.api.routes.search` / `.answer` - the
names the route actually resolves - so nothing here touches Postgres, torch, or
a paid API. What is being tested is the wiring: status codes, response shape, and
that internal errors never reach the client as internals.
"""

import pytest

from app.generation.llm import IDK, LLMError
from tests.conftest import make_hit

HITS = [make_hit(973, 693, 3.53, "If the engine will not start, check..."),
        make_hit(975, 694, 1.20, "The starter motor turns over slowly...")]


@pytest.fixture
def stub_rag(monkeypatch):
    """Install fake retrieval and generation. Returns a call-recording dict."""
    calls = {}

    def install(hits=HITS, answer_text="Check the battery (p. 693).", answer_exc=None):
        def fake_search(question, k):
            calls["search"] = (question, k)
            return hits

        def fake_answer(question, chunks):
            calls["answer"] = (question, chunks)
            if answer_exc:
                raise answer_exc
            return answer_text

        monkeypatch.setattr("app.api.routes.search", fake_search)
        monkeypatch.setattr("app.api.routes.answer", fake_answer)
        return calls

    return install


def test_ask_returns_answer_with_page_citations(client, stub_rag):
    stub_rag()
    r = client.post("/api/ask", json={"question": "engine won't start", "k": 2})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Check the battery (p. 693)."
    assert body["refused"] is False
    assert [s["page"] for s in body["sources"]] == [693, 694]
    assert [s["chunk_id"] for s in body["sources"]] == [973, 975]
    # The UI shows these; they must be present and ordered retrieval <= total.
    assert body["retrieval_ms"] <= body["total_ms"]
    assert body["strategy"] in {"dense", "hybrid", "reranked"}


def test_k_is_passed_through_to_retrieval(client, stub_rag):
    calls = stub_rag()
    client.post("/api/ask", json={"question": "brake fluid type", "k": 3})
    assert calls["search"] == ("brake fluid type", 3)


def test_retrieved_chunks_are_what_the_llm_sees(client, stub_rag):
    # Grounding is the whole premise: the model must be handed the retrieved
    # chunks, not the raw question alone.
    calls = stub_rag()
    client.post("/api/ask", json={"question": "engine won't start"})
    assert calls["answer"][1] == HITS


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                      # below min_length=3
    {"question": "x" * 501},                 # above max_length=500
    {"question": "valid question", "k": 0},  # k below ge=1
    {"question": "valid question", "k": 99}, # k above le=20
    {},                                      # question missing entirely
])
def test_bad_input_is_rejected_before_any_work_happens(client, payload):
    # 422 from pydantic, so a malformed request never costs an embedding pass or
    # an LLM call.
    assert client.post("/api/ask", json=payload).status_code == 422


def test_empty_retrieval_refuses_instead_of_guessing(client, monkeypatch):
    # Note: only `search` is stubbed. The REAL answer() runs and short-circuits
    # on empty chunks, so this also proves it never calls the provider.
    monkeypatch.setattr("app.api.routes.search", lambda q, k: [])
    body = client.post("/api/ask", json={"question": "how do I bake bread"}).json()

    assert body["answer"] == IDK
    assert body["refused"] is True
    assert body["sources"] == []


def test_llm_failure_becomes_502_without_leaking_details(client, stub_rag):
    stub_rag(answer_exc=LLMError("provider returned 429 for key sk-abc123"))
    r = client.post("/api/ask", json={"question": "engine won't start"})

    assert r.status_code == 502
    assert r.json()["detail"] == "language model unavailable"
    assert "sk-abc123" not in r.text        # the key never reaches the client


def test_retrieval_failure_becomes_503_without_leaking_details(client, monkeypatch):
    def boom(question, k):
        raise RuntimeError("connection to mechanic:mechanic@localhost:5433 refused")

    monkeypatch.setattr("app.api.routes.search", boom)
    r = client.post("/api/ask", json={"question": "engine won't start"})

    assert r.status_code == 503
    assert r.json()["detail"] == "retrieval unavailable"
    assert "5433" not in r.text            # no DSN in the response body


def test_request_id_is_echoed_for_correlation(client, stub_rag):
    stub_rag()
    r = client.post("/api/ask", json={"question": "engine won't start"},
                    headers={"X-Request-ID": "trace-me-42"})
    # A caller-supplied id survives, so a frontend or load balancer trace joins
    # up with the server logs for the same request.
    assert r.headers["X-Request-ID"] == "trace-me-42"

    generated = client.post("/api/ask", json={"question": "engine won't start"})
    assert generated.headers["X-Request-ID"]      # one is minted when absent
