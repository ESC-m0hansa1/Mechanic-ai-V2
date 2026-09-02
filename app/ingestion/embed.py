"""Turn text into 384-dim vectors, with ONNX Runtime rather than PyTorch.

This used to be four lines around SentenceTransformer. It is longer now because
torch and transformers cost ~356 MB of resident memory purely to import, and the
deploy target caps a service at 512 MB - the reasoning, the measurements and the
two behaviours that had to be reproduced by hand (CLS pooling, L2 normalisation)
are all in app/core/onnx_backend.py.

The public surface is deliberately unchanged: embed_texts() takes the same
arguments and returns the same lists, so dense.py and store.py never learned
that the backend was swapped.
"""

from app.core.config import settings
from app.core.onnx_backend import OnnxEmbedder

_model: OnnxEmbedder | None = None       # module-level cache (lazy singleton)


def get_model() -> OnnxEmbedder:
    """Load the model once, then reuse it (loading is slow; do it a single time)."""
    global _model                        # we're assigning to the module-level name
    if _model is None:
        # Lazy for the same reason as before, plus a new one: the weights are a
        # build artifact now, not a download, so an import-time load would make
        # every test and CLI require that scripts/export_onnx.py had been run.
        _model = OnnxEmbedder(
            f"{settings.onnx_model_dir}/embedder",
            intra_threads=settings.onnx_intra_threads,
        )
    return _model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Turn a list of strings into a list of 384-dim vectors.

    Vectors are unit length, so cosine similarity == dot product - which is what
    both the pgvector query and the stored embeddings assume.
    """
    # show_progress is accepted and ignored: it existed because
    # SentenceTransformer.encode drew a tqdm bar, and ingestion passed False to
    # keep it out of server logs. Kept in the signature so the ingestion call
    # site did not have to change; there is no bar to suppress any more.
    del show_progress
    vectors = get_model().encode(texts, batch_size=32)
    return vectors.tolist()              # numpy array -> plain Python lists (for DB insert)


if __name__ == "__main__":
    import numpy as np
    samples = [
        "How do I check the engine oil level?",
        "Checking engine oil: park on level ground and read the dipstick.",
        "The audio system supports Bluetooth pairing.",
    ]
    vecs = embed_texts(samples)
    print(f"count: {len(vecs)}, dim: {len(vecs[0])}")   # expect dim: 384
    a, b, c = np.array(vecs)
    print(f"sim(oil-Q, oil-A):    {a @ b:.3f}")          # related -> HIGH
    print(f"sim(oil-Q, audio):    {a @ c:.3f}")          # unrelated -> LOW
