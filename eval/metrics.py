"""Retrieval metrics - pure functions, no I/O, so they are trivially testable.

Deliberately hand-written rather than pulled from a library: in an interview the
question is never "which package computes MRR", it is "what does MRR mean and
why did you pick it". These are the definitions this project uses.
"""


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """1.0 if ANY relevant chunk appears in the top k, else 0.0.

    The user-facing question: did we put at least one useful passage in front of
    the model? Averaged over the golden set this is "hit rate", the metric that
    correlates most directly with "did the answer come out right".
    """
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Fraction of ALL relevant chunks that made it into the top k.

    Stricter than hit_at_k and reported separately: a question with 3 relevant
    chunks where we found 1 scores hit=1.0 but recall=0.33. Reporting only the
    first would flatter the system.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """1 / (rank of the first relevant hit), 1-indexed; 0.0 if none was found.

    Rewards putting the right chunk FIRST, not merely somewhere in the list.
    That matters here because the LLM's context is ordered and a top-1 hit
    survives any later truncation of the prompt.
    """
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). p95 = "95% of requests beat this".

    Averages hide the slow tail that users actually complain about, so latency
    is reported as p50/p95 rather than as a mean.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    # ceil(p/100 * n) - 1, clamped: index of the first value at or above the rank
    idx = max(0, min(len(ordered) - 1, int(round(p / 100 * len(ordered) + 0.5)) - 1))
    return ordered[idx]
