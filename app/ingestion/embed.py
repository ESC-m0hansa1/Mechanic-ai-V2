from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, 512-token window, retrieval-tuned
_model = None                            # module-level cache (lazy singleton)


def get_model() -> SentenceTransformer:
    """Load the model once, then reuse it (loading is slow; do it a single time)."""
    global _model                        # we're assigning to the module-level name
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)  # 1st call downloads (~130MB), then cached on disk
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into a list of 384-dim vectors."""
    model = get_model()
    # normalize_embeddings=True -> unit-length vectors, so cosine similarity == dot product
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
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