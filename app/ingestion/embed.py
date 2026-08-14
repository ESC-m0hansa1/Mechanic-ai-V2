from sentence_transformers import SentenceTransformer

from app.core.config import settings

_model = None                            # module-level cache (lazy singleton)


def get_model() -> SentenceTransformer:
    """Load the model once, then reuse it (loading is slow; do it a single time)."""
    global _model                        # we're assigning to the module-level name
    if _model is None:
        # 1st call downloads (~130MB), then it's cached on disk.
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    """Turn a list of strings into a list of 384-dim vectors."""
    model = get_model()
    # normalize_embeddings=True -> unit-length vectors, so cosine similarity == dot product
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=show_progress,  # off by default: progress bars pollute server logs
    )
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