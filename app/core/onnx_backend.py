"""ONNX Runtime inference: the two transformer models without PyTorch.

Why not sentence-transformers
-----------------------------
Measured resident memory of this server, loading the same two checkpoints:

    baseline python                14 MB
    + torch                       200 MB
    + sentence_transformers       370 MB
    + bge-small loaded            482 MB
    + cross-encoder loaded        509 MB

The free tier caps a service at 512 MB. Only ~139 MB of that is model weights -
the other ~356 MB is torch and transformers merely being imported. So the fix was
not a smaller model or dropping the reranker (482 MB is already spent once the
embedder loads); it was deleting the framework. ONNX Runtime replaces it with a
~40 MB dependency and no Python-level model code.

What that costs
---------------
sentence-transformers did tokenization, pooling and normalisation for us. Here
they are explicit, and they have to match the originals exactly or retrieval
silently degrades - the thresholds in eval/ are calibrated against the torch
scores. The two behaviours were read off the loaded torch models rather than
assumed (see scripts/check_parity.py, which asserts the port is faithful):

  * bge-small-en-v1.5 pools with the **CLS token**, not the mean, then L2
    normalises. Mean pooling here would produce plausible-looking vectors with
    quietly worse ranking - the failure mode that motivated a parity check.
  * ms-marco-MiniLM-L-6-v2 has num_labels=1 and an **Identity** activation, so
    its output is a raw logit in roughly [-11, +11] - not a probability. The
    calibrated refusal threshold (-3.60) only means anything on that scale.

Weights are fp32, matching torch arithmetic. int8 is available from the export
script but is lossy and would need the eval re-run before it could be trusted.
"""

import json
import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class OnnxTransformer:
    """A tokenizer plus an ONNX graph, loaded once and reused.

    Subclasses decide what to do with the raw graph output; this class owns only
    the parts that are identical for both models.
    """

    def __init__(self, model_dir: str | Path, intra_threads: int = 0):
        self.dir = Path(model_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(
                f"no ONNX model at {self.dir}. Run: python scripts/export_onnx.py"
            )
        self._intra_threads = intra_threads

    @cached_property
    def _meta(self) -> dict:
        return json.loads((self.dir / "runtime.json").read_text())

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        # Truncate at the window the graph was traced for. Without this a long
        # chunk would produce a sequence the model was never trained on.
        tok.enable_truncation(max_length=self._meta["max_length"])
        # Pad to the longest item in each batch, not to max_length: the sequence
        # axis is dynamic, so a batch of short queries costs a short forward pass.
        tok.enable_padding(pad_id=tok.token_to_id("[PAD]") or 0, pad_token="[PAD]")
        return tok

    @cached_property
    def _session(self) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        if self._intra_threads:
            # The deploy target allocates a fraction of a core. Letting ORT spawn
            # one thread per visible CPU there means threads contending for a
            # slice none of them can fill, which is slower than staying single
            # threaded. 0 keeps ORT's own default, which is right on a real box.
            opts.intra_op_num_threads = self._intra_threads
        # ORT pre-allocates a reusable memory arena per session, trading memory
        # for allocator speed. That default is wrong here: measured across both
        # sessions it cost 143 MB to save 4 ms on an 8-pair rerank, and 143 MB is
        # 28% of the 512 MB budget. Outputs are bit-identical either way - the
        # arena is an allocation strategy, not arithmetic.
        opts.enable_cpu_mem_arena = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess = ort.InferenceSession(
            str(self.dir / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("onnx session ready: %s (threads=%s)",
                    self._meta["source_model"], self._intra_threads or "auto")
        return sess

    def _forward(self, encoded) -> np.ndarray:
        """Tokenize a batch and run one forward pass, returning the raw output."""
        encs = self._tokenizer.encode_batch(encoded)
        # The graph declares exactly which inputs it wants (recorded at export).
        # Feeding an input the graph does not have raises at run time, so we
        # intersect rather than assume all three BERT inputs are present.
        feed = {
            "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
        }
        wanted = {i.name for i in self._session.get_inputs()}
        feed = {k: v for k, v in feed.items() if k in wanted}
        return self._session.run(None, feed)[0]

    def _batched(self, items: list, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]


class OnnxEmbedder(OnnxTransformer):
    """bge-small-en-v1.5: CLS-pooled, L2-normalised 384-dim sentence vectors."""

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for batch in self._batched(texts, batch_size):
            hidden = self._forward(batch)          # (batch, seq, 384)
            pooled = hidden[:, 0]                  # CLS token == position 0
            # L2 normalise, so cosine similarity is a plain dot product - which
            # is what the pgvector query and the stored vectors both assume.
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append(pooled / np.maximum(norms, 1e-12))
        return np.vstack(out).astype(np.float32)


class OnnxCrossEncoder(OnnxTransformer):
    """ms-marco-MiniLM-L-6-v2: one raw relevance logit per (query, chunk) pair.

    Exposes `.predict(pairs, **kwargs)` so it is a drop-in for the
    sentence-transformers CrossEncoder this replaced - rerank() and the stub in
    tests/test_rerank.py both keep working untouched.
    """

    def predict(self, pairs, batch_size: int = 16, **_ignored) -> np.ndarray:
        out = []
        for batch in self._batched(list(pairs), batch_size):
            # tokenizers takes (text, pair) tuples and sets token_type_ids to
            # 0 for the query and 1 for the chunk, which is what makes this a
            # cross-encoder rather than two concatenated strings.
            logits = self._forward([tuple(p) for p in batch])   # (batch, 1)
            out.append(logits[:, 0])
        return np.concatenate(out).astype(np.float64)
