"""Shared fixtures.

The design rule for this test suite: **no test requires the database, the
embedding model, the cross-encoder, or the LLM provider.** Those are all real
dependencies of the running app, and each one would make the suite slow, or
flaky, or cost money - which is how a suite stops being run at all.

So the seams are stubbed at the boundary the route actually calls:
`app.api.routes.search` and `app.api.routes.answer`. Everything below those two
names is tested separately as pure functions (fusion maths, metrics, the noise
heuristic, the rerank ordering), and the one test that genuinely needs live
infrastructure is marked `integration` so `-m "not integration"` skips it.
"""

import os

import pytest

# Settings are read at import time and APP_NAME/DATABASE_URL are required, so
# they must exist before app.core.config is imported. setdefault, not assignment:
# a developer running against their own .env should keep their values.
os.environ.setdefault("APP_NAME", "Mechanic AI (test)")
os.environ.setdefault(
    "DATABASE_URL", "postgresql://mechanic:mechanic@localhost:5433/mechanic"
)

from fastapi.testclient import TestClient      # noqa: E402  (after env setup)

from app.main import app                       # noqa: E402


def make_hit(chunk_id: int, page: int, score: float, content: str = "text") -> dict:
    """A retriever result. Every strategy returns exactly this shape."""
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {"page": page, "local_index": 0},
        "score": score,
    }


@pytest.fixture
def client() -> TestClient:
    """A test client that does NOT run the lifespan handler.

    Starlette only runs startup/shutdown when TestClient is used as a context
    manager. Skipping it is the point: lifespan calls warm_up(), which loads two
    transformer models and hits Postgres. These tests exercise the routes, not
    the boot sequence.
    """
    return TestClient(app)
