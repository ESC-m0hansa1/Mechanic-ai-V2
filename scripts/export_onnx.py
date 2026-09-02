"""Export the two models to ONNX so the server does not need PyTorch.

Why this exists
---------------
Measured on this machine, the server's resident memory before any request:

    baseline python                14 MB
    + torch                       200 MB
    + sentence_transformers       370 MB
    + bge-small loaded            482 MB
    + cross-encoder loaded        509 MB

The free tier we deploy to caps a service at 512 MB, so the app OOM-killed with
3 MB to spare. Note where the weight actually is: only ~139 MB of that is model
parameters. The other ~356 MB is torch and transformers merely being *imported*.
Dropping to dense-only retrieval would not have helped - 482 MB is already spent
once the embedding model loads - so the reranker was never the problem. Torch
was. ONNX Runtime is the same two models with a ~40 MB runtime instead.

This script runs at BUILD time (or by hand), never in the server. It is the only
place torch is still required, which is why torch is not in requirements.txt.

Exported weights are fp32, deliberately
---------------------------------------
int8 dynamic quantization would be ~4x smaller again, but quantization is lossy:
it perturbs the scores the retrieval thresholds are calibrated against. fp32 ONNX
is the same arithmetic as torch, so ranking is reproducible. Pass --int8 to also
emit quantized copies, but re-run the eval before trusting them - see
scripts/check_parity.py.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Kept in sync with app/core/config.py defaults. Passed explicitly rather than
# imported so this script does not need a .env to run in a build stage.
EMBEDDER = "BAAI/bge-small-en-v1.5"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OPSET = 17


def _export(model, out_dir: Path, output_name: str, max_len: int) -> Path:
    """Trace `model` to ONNX with batch and sequence length left dynamic.

    Both dimensions must be dynamic: batch because ingestion embeds 32 chunks at
    a time while a query is 1, and sequence because padding to a fixed 512 would
    make every short query cost a full-length forward pass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    model.eval()

    # A real tokenized batch as the tracing example. Shape (2, 16) so the tracer
    # cannot silently bake in batch=1 or a squeezed dimension.
    dummy = {
        "input_ids": torch.ones(2, 16, dtype=torch.long),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 16, dtype=torch.long),
    }
    names = list(dummy)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),                       # dict-as-kwargs: matches BERT's forward()
            str(path),
            input_names=names,
            output_names=[output_name],
            dynamic_axes={n: {0: "batch", 1: "seq"} for n in names}
            | {output_name: {0: "batch", 1: "seq"}},
            opset_version=OPSET,
            do_constant_folding=True,
            # torch >= 2.6 defaults to the dynamo exporter, which needs onnxscript
            # and takes `dynamic_shapes` instead of `dynamic_axes`. The legacy
            # TorchScript path is explicitly requested because it is the one that
            # has traced BERT encoders reliably for years, and this graph is
            # verified numerically afterwards (scripts/check_parity.py) either way.
            dynamo=False,
        )

    # The tokenizer must ship next to the weights: at runtime we load it with the
    # `tokenizers` library (no transformers), which needs tokenizer.json. Calling
    # save_pretrained on the *fast* tokenizer is what guarantees that file exists
    # - ms-marco ships only vocab.txt on the Hub.
    tok = AutoTokenizer.from_pretrained(model.name_or_path, use_fast=True)
    tok.save_pretrained(out_dir)
    assert (out_dir / "tokenizer.json").exists(), "fast tokenizer did not emit tokenizer.json"

    # Runtime needs to know the truncation limit and which inputs the graph wants;
    # recording it here means the server never re-derives it from a config it
    # would otherwise need transformers to parse.
    (out_dir / "runtime.json").write_text(json.dumps({
        "max_length": max_len,
        "input_names": names,
        "output_name": output_name,
        "source_model": model.name_or_path,
        "opset": OPSET,
    }, indent=2))

    mb = path.stat().st_size / 1e6
    print(f"  {model.name_or_path} -> {path}  ({mb:.0f} MB)")
    return path


def _quantize(src: Path) -> None:
    """Emit an int8 dynamic-quantized sibling next to `src`."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst = src.with_name("model.int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"  int8: {dst}  ({dst.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/onnx", type=Path)
    ap.add_argument("--int8", action="store_true", help="also emit int8 copies (lossy)")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)     # stale graphs are worse than missing ones

    print("exporting embedder (CLS-pooled, then L2-normalised at runtime)")
    emb = _export(AutoModel.from_pretrained(EMBEDDER),
                  args.out / "embedder", "last_hidden_state", 512)

    print("exporting cross-encoder (single relevance logit)")
    rer = _export(AutoModelForSequenceClassification.from_pretrained(RERANKER),
                  args.out / "reranker", "logits", 512)

    if args.int8:
        print("quantizing (lossy - re-run scripts/check_parity.py)")
        _quantize(emb)
        _quantize(rer)

    print(f"\ndone -> {args.out}")
    print("verify with: python scripts/check_parity.py")


if __name__ == "__main__":
    main()
