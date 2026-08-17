"""Offline retrieval evaluation: the numbers that go in the README table.

Retrieval-only by default - it never calls the LLM, so a full run is free and
takes seconds. That matters: an eval you avoid running because it costs money
stops being run at all, and then the metrics table rots.

    python -m eval.run_eval                 # current RETRIEVAL_STRATEGY
    python -m eval.run_eval --strategy hybrid
    python -m eval.run_eval --all           # every registered strategy, one table
"""

import argparse
import json
import time
from pathlib import Path

from app.core.config import settings
from app.retrieval.search import STRATEGIES, search
from eval.metrics import hit_at_k, percentile, recall_at_k, reciprocal_rank

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET.read_text(encoding="utf-8"))


def calibrate_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """Find the top-1 score cutoff that best separates answerable from not.

    Learned from the data instead of guessed: an earlier hand-picked 0.5 cutoff
    was useless here because an off-topic question still scored 0.54 while a
    good hit scored 0.80. Sweeping every observed score and keeping the best
    split also reports HOW separable the two groups are - if accuracy is barely
    above chance, no threshold will save us and refusal has to come from the
    model reading the excerpts instead.
    """
    if not answerable or not unanswerable:
        return {"refusal_threshold": None, "refusal_accuracy": None}
    best = (0.0, None)
    for cut in sorted(set(answerable + unanswerable)):
        # Predict "answerable" when top-1 score >= cut.
        correct = sum(s >= cut for s in answerable) + sum(s < cut for s in unanswerable)
        accuracy = correct / (len(answerable) + len(unanswerable))
        if accuracy > best[0]:
            best = (accuracy, cut)
    return {
        "refusal_threshold": round(best[1], 4) if best[1] is not None else None,
        "refusal_accuracy": round(best[0], 3),
        "top1_answerable_mean": round(sum(answerable) / len(answerable), 4),
        "top1_unanswerable_mean": round(sum(unanswerable) / len(unanswerable), 4),
    }


def evaluate(strategy: str, k: int, cases: list[dict]) -> dict:
    """Run every golden question through `strategy` and aggregate the metrics."""
    # search() dispatches on this setting, so flipping it here lets one process
    # measure every strategy without reloading the embedding model each time.
    settings.retrieval_strategy = strategy

    hits, recalls, rrs, latencies = [], [], [], []
    hits_at_1, hits_at_3 = [], []
    top1_answerable, top1_unanswerable = [], []

    # Throwaway query first: the embedding model loads lazily, so without this
    # the very first timing measures a ~2-6 s model load and the latency numbers
    # describe a cold start rather than steady-state serving. The app does the
    # same thing in its startup hook.
    search("warm up the embedding model", 1)

    for case in cases:
        relevant = set(case["relevant_chunk_ids"])
        t0 = time.perf_counter()
        results = search(case["question"], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [r["id"] for r in results]
        top1 = float(results[0]["score"]) if results else 0.0

        if not relevant:
            # Unanswerable: ranking metrics are undefined (no chunk is correct),
            # so this case only contributes to threshold calibration.
            top1_unanswerable.append(top1)
            continue

        top1_answerable.append(top1)
        hits_at_1.append(hit_at_k(retrieved, relevant, 1))
        hits_at_3.append(hit_at_k(retrieved, relevant, 3))
        hits.append(hit_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))

    n = len(hits) or 1
    return {
        "strategy": strategy,
        "k": k,
        "answerable_questions": len(hits),
        "unanswerable_questions": len(top1_unanswerable),
        # hit@1 and hit@3 are reported because hit@5 saturates at 1.0 on this
        # golden set - a ceiling metric cannot show whether reranking helped.
        "hit_at_1": round(sum(hits_at_1) / n, 3),
        "hit_at_3": round(sum(hits_at_3) / n, 3),
        "hit_at_k": round(sum(hits) / n, 3),
        "recall_at_k": round(sum(recalls) / n, 3),
        "mrr": round(sum(rrs) / n, 3),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        **calibrate_threshold(top1_answerable, top1_unanswerable),
    }



def print_table(rows: list[dict]) -> None:
    """Emit markdown so the README table is copy-pasted, never hand-typed."""
    print()
    print("| strategy | hit@1 | hit@3 | hit@5 | recall@5 | MRR | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['strategy']} | {r['hit_at_1']} | {r['hit_at_3']} | "
              f"{r['hit_at_k']} | {r['recall_at_k']} | {r['mrr']} | "
              f"{r['latency_p50_ms']} | {r['latency_p95_ms']} |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies.")
    parser.add_argument("--strategy", default=None, choices=sorted(STRATEGIES))
    parser.add_argument("--all", action="store_true", help="evaluate every strategy")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cases = load_golden_set()
    strategies = (
        sorted(STRATEGIES) if args.all else [args.strategy or settings.retrieval_strategy]
    )

    print(f"golden set: {len(cases)} questions | k={args.k}")
    rows = []
    for name in strategies:
        row = evaluate(name, args.k, cases)
        rows.append(row)
        RESULTS_DIR.mkdir(exist_ok=True)
        # One file per strategy: the README numbers stay reproducible and diffable.
        (RESULTS_DIR / f"{name}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(f"  {name:<9} hit@1={row['hit_at_1']:<6} hit@{args.k}={row['hit_at_k']:<6} "
              f"recall@{args.k}={row['recall_at_k']:<6} mrr={row['mrr']:<6} "
              f"p95={row['latency_p95_ms']}ms  "
              f"refusal_acc={row['refusal_accuracy']} @cut={row['refusal_threshold']}")
    print_table(rows)


if __name__ == "__main__":
    main()
