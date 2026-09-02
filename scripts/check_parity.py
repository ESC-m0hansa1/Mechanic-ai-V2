"""Prove the ONNX port produces the same numbers as PyTorch did.

Swapping inference runtimes is exactly the kind of change that appears to work -
the app boots, answers look sane - while quietly ranking worse, because nothing
crashes if you mean-pool a model that expects CLS pooling or forget to normalise.
The retrieval metrics in the README and the calibrated refusal thresholds in
eval/ were measured on the torch stack, so they are only still valid if the port
is numerically faithful. This asserts that, rather than trusting it.

Requires torch (dev only). Run after scripts/export_onnx.py:

    python scripts/check_parity.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.onnx_backend import OnnxCrossEncoder, OnnxEmbedder  # noqa: E402

EMBEDDER = "BAAI/bge-small-en-v1.5"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Deliberately mixed: the bge query prefix, manual jargon, a long chunk that will
# be truncated, and a short query - so padding and truncation are both exercised.
TEXTS = [
    "Represent this sentence for searching relevant passages: how do I check the engine oil level?",
    "Checking engine oil: park the vehicle on level ground, stop the engine and wait five minutes.",
    "The audio system supports Bluetooth pairing.",
    "ISOFIX child restraint anchor locations and AdBlue refill intervals for the 1GD-FTV engine.",
    "oil",
    " ".join(["Tighten the wheel nuts to the specified torque in the order shown."] * 40),
]
PAIRS = [
    ("my truck won't turn over", "If the engine will not start, check the battery terminals first."),
    ("my truck won't turn over", "Bluetooth audio pairing is described on the following page."),
    ("ISOFIX anchor", "ISOFIX lower anchors are located at the rear seat cushion."),
    ("how much AdBlue", " ".join(["Refill the AdBlue tank when the warning appears."] * 40)),
]

# Tolerances: fp32 ONNX is the same arithmetic as torch, but graph optimisation
# reorders operations, so bit-identical is the wrong bar. 1e-4 on a unit vector
# is far below the gap between any two competing chunks.
EMB_TOL = 1e-4
LOGIT_TOL = 2e-3


def main() -> int:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    failures = []

    print("embedder: torch vs onnx")
    t_emb = SentenceTransformer(EMBEDDER).encode(TEXTS, normalize_embeddings=True)
    o_emb = OnnxEmbedder("models/onnx/embedder").encode(TEXTS)
    dev = np.abs(t_emb - o_emb).max()
    cos = np.einsum("ij,ij->i", t_emb, o_emb)          # both are unit length
    print(f"  max abs deviation {dev:.2e}   min cosine {cos.min():.8f}")
    if dev > EMB_TOL:
        failures.append(f"embedding deviation {dev:.2e} > {EMB_TOL:.0e}")
    if not np.isclose(np.linalg.norm(o_emb, axis=1), 1.0, atol=1e-5).all():
        failures.append("onnx embeddings are not unit length (normalisation lost)")

    print("cross-encoder: torch vs onnx")
    t_sc = np.asarray(CrossEncoder(RERANKER, max_length=512)
                      .predict(PAIRS, batch_size=16, show_progress_bar=False), dtype=np.float64)
    o_sc = OnnxCrossEncoder("models/onnx/reranker").predict(PAIRS, batch_size=16)
    dev = np.abs(t_sc - o_sc).max()
    print(f"  torch {np.round(t_sc, 4).tolist()}")
    print(f"  onnx  {np.round(o_sc, 4).tolist()}")
    print(f"  max abs deviation {dev:.2e}")
    if dev > LOGIT_TOL:
        failures.append(f"logit deviation {dev:.2e} > {LOGIT_TOL:.0e}")
    # The scale matters as much as the values: the refusal threshold (-3.60) is
    # calibrated on raw logits. A sigmoid slipping in would squash everything
    # into [0,1] and the threshold would silently accept every question.
    if o_sc.min() > 0.0 or o_sc.max() < 1.0:
        failures.append(f"logits look squashed into [0,1] - activation is not Identity: {o_sc}")

    # What actually has to hold: the same ordering. Deviations only matter if
    # they reorder chunks, so check that directly rather than inferring it.
    print("ranking parity")
    if (np.argsort(-t_sc) == np.argsort(-o_sc)).all():
        print("  rank order identical")
    else:
        failures.append("cross-encoder rank order differs between torch and onnx")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS - the port is numerically faithful; measured metrics still apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
