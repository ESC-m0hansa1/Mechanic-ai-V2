"""The metric definitions are hand-written (eval/metrics.py), so they are tested.

Each test pins one property of the definition rather than one example, because
the examples are what a reader would check by eye anyway; the properties are
what silently break. hit@k ignores multiplicity, recall@k does not; MRR looks at
the FIRST relevant result and nothing else.
"""

from eval.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank


def test_hit_at_k_is_binary_not_a_count():
    # Two relevant chunks in the top 3 is still a hit of 1.0, not 2.0.
    assert hit_at_k([1, 2, 3], {2, 3}, 3) == 1.0
    assert hit_at_k([1, 2, 3], {2}, 3) == 1.0


def test_hit_at_k_respects_the_cutoff():
    # The relevant chunk is at rank 3, so it counts at k=3 but not at k=2.
    assert hit_at_k([9, 8, 1], {1}, 3) == 1.0
    assert hit_at_k([9, 8, 1], {1}, 2) == 0.0


def test_recall_at_k_is_the_fraction_of_relevant_found():
    # One of two found -> 0.5. This is the metric that punishes a strategy for
    # surfacing only one of the two pages an answer is split across.
    assert recall_at_k([1, 5, 6], {1, 2}, 3) == 0.5
    assert recall_at_k([1, 2, 6], {1, 2}, 3) == 1.0


def test_recall_at_k_with_no_relevant_chunks_is_zero_not_a_crash():
    # Unanswerable questions have an empty relevant set. The harness skips them,
    # but the function must not divide by zero if it is called anyway.
    assert recall_at_k([1, 2], set(), 2) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_hit_only():
    assert reciprocal_rank([7, 1], {1}) == 0.5          # rank 2 -> 1/2
    assert reciprocal_rank([1, 7], {1}) == 1.0          # rank 1 -> 1/1
    # A second relevant hit further down changes nothing: MRR measures how far
    # the user has to read before the FIRST useful result.
    assert reciprocal_rank([7, 1, 2], {1, 2}) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_retrieved():
    assert reciprocal_rank([4, 5, 6], {1}) == 0.0


def test_percentile_picks_an_actual_observed_value():
    # Nearest-rank, so p50 of ten samples is a real measurement, not an
    # interpolation between two of them. Latency numbers you can point at.
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) in values
    assert percentile(values, 100) == 100
    assert percentile(values, 0) == 10


def test_percentile_of_nothing_is_zero():
    # An all-unanswerable run collects no latencies; reporting must not explode.
    assert percentile([], 95) == 0.0
