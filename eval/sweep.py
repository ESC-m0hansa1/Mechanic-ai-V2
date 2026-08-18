"""Sweep one integer setting and report what it does to the metrics.

Every tuning claim in this repo came from here rather than from a blog post:

    python -m eval.sweep --strategy reranked --setting candidate_pool --values 10 15 20 30
    python -m eval.sweep --strategy hybrid   --setting fusion_depth   --values 5 10 20 30

Two honest caveats that belong in the README next to any number this prints:

* 24 answerable questions is a small sample. A 0.02 MRR difference between two
  values is inside the noise; only differences of a few tenths mean anything.
* Tuning on the same 24 questions that produce the headline metrics is
  overfitting. The defence is that each knob is a single integer with a
  monotone-ish effect, not a free parameter vector - but it is still tuning on
  the test set, and it should be said out loud rather than hidden.
"""

import argparse

from app.core.config import settings
from app.retrieval.search import STRATEGIES
from eval.run_eval import evaluate, load_golden_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep one setting over values.")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--setting", required=True,
                        help="attribute on settings, e.g. candidate_pool")
    parser.add_argument("--values", required=True, type=int, nargs="+")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if not hasattr(settings, args.setting):
        raise SystemExit(f"no such setting: {args.setting}")

    cases = load_golden_set()
    original = getattr(settings, args.setting)

    print(f"\n| {args.setting} | hit@1 | hit@3 | recall@{args.k} | MRR | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|")
    try:
        for value in args.values:
            # Mutating the shared settings object is safe here because this is a
            # single-threaded offline script; the app never rebinds settings.
            setattr(settings, args.setting, value)
            row = evaluate(args.strategy, args.k, cases)
            print(f"| {value} | {row['hit_at_1']} | {row['hit_at_3']} | "
                  f"{row['recall_at_k']} | {row['mrr']} | "
                  f"{row['latency_p50_ms']} | {row['latency_p95_ms']} |")
    finally:
        setattr(settings, args.setting, original)
    print()


if __name__ == "__main__":
    main()
