"""Retrieval strategy dispatcher.

The route never imports a specific retriever - it calls `search()`, which
dispatches on the RETRIEVAL_STRATEGY setting. That means the
baseline -> hybrid -> reranked progression is a config change, not a code
change, and the eval harness can measure every strategy through one interface.
"""

from app.core.config import settings
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search

# Populated as strategies land: dense (C4), hybrid (C6), reranked (C7).
STRATEGIES = {
    "dense": dense_search,      # baseline: vector similarity only
    "hybrid": hybrid_search,    # dense + BM25 fused with Reciprocal Rank Fusion
}


def search(query: str, k: int = 5) -> list[dict]:
    """Retrieve the top-k chunks using the configured strategy."""
    try:
        strategy = STRATEGIES[settings.retrieval_strategy]
    except KeyError:
        raise ValueError(
            f"unknown RETRIEVAL_STRATEGY={settings.retrieval_strategy!r}; "
            f"expected one of {sorted(STRATEGIES)}"
        )
    return strategy(query, k)
