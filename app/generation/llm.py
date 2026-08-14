"""Answer generation: turn retrieved chunks + a question into a grounded answer.

Provider-agnostic on purpose. Mistral, Groq, Together, vLLM and Ollama all
speak the same OpenAI-compatible `/chat/completions` shape, so switching
provider is three env vars (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY), not a
code change.
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# The refusal string is a constant because the eval harness and tests assert on
# it - a typo here would silently break "I don't know" detection.
IDK = "I don't know based on the manual."

SYSTEM_PROMPT = (
    "You are a vehicle repair assistant. Answer ONLY using the manual excerpts "
    "provided. Cite the page for each fact like (p. 567). Be concise. "
    f"If the excerpts do not contain the answer, reply exactly: '{IDK}'"
)


class LLMError(RuntimeError):
    """Raised when the language model cannot be reached or returns junk.

    A dedicated exception type lets the API layer map this to a 502 without
    catching unrelated bugs in our own code.
    """


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Stitch retrieved chunks + the question into one user message."""
    context = "\n\n".join(
        f"[page {c['metadata']['page']}] {c['content']}" for c in chunks
    )
    return f"MANUAL EXCERPTS:\n{context}\n\nQUESTION: {question}"


def answer(question: str, chunks: list[dict]) -> str:
    """Ask the LLM to answer `question` using only `chunks`."""
    if not chunks:
        # Retrieval found nothing -> there is nothing to ground an answer in.
        # Short-circuit instead of paying for a call that can only hallucinate.
        return IDK

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        "max_tokens": settings.llm_max_tokens,
        "temperature": 0.0,  # deterministic-ish: required for repeatable evals
    }

    try:
        resp = httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json=payload,
            timeout=settings.llm_timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as exc:  # 4xx/5xx from the provider
        raise LLMError(f"provider returned {exc.response.status_code}") from exc
    except httpx.RequestError as exc:  # DNS, TLS, timeout, connection refused
        raise LLMError(f"could not reach provider: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:  # unexpected JSON shape
        raise LLMError(f"unexpected provider response: {exc}") from exc


if __name__ == "__main__":
    from app.retrieval.search import search

    q = "how do I check the engine oil level?"
    print(answer(q, search(q, k=5)))
