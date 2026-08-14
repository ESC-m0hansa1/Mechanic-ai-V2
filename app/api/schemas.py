"""Request/response shapes for the API.

Pydantic models here do double duty: they validate incoming JSON (rejecting bad
input with a 422 before our code runs) and they document the API automatically
in /docs.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """What a client must POST to /ask."""

    # min_length stops empty questions; max_length stops abuse / runaway prompts.
    question: str = Field(min_length=3, max_length=500)
    # ge/le = "greater or equal" / "less or equal" -> k must be 1..20.
    k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    """One retrieved manual chunk, returned so the answer is auditable."""

    chunk_id: int
    page: int
    score: float
    preview: str


class AskResponse(BaseModel):
    """What /ask returns."""

    question: str
    answer: str
    sources: list[SourceChunk]
    retrieval_ms: int
    total_ms: int
    strategy: str
