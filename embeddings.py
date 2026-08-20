"""
embeddings.py — shared embedding utilities.

Uses sentence-transformers running locally (no API calls, no rate limits,
no daily quota — the whole point after today's Gemini quota fight).

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~80MB, downloads once from Hugging Face on first run, then cached
    locally (default cache: ~/.cache/huggingface/) — every run after the
    first is fully offline
  - Small and fast, but strong for short-sentence similarity, which is
    exactly what your captions and (presumably short) blog excerpts are
"""

import json
import math

from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model() -> SentenceTransformer:
    """Lazy singleton — load the model once per process, not once per call."""
    global _MODEL
    if _MODEL is None:
        print("Loading embedding model (first run downloads ~80MB, then cached)...")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text: str) -> list:
    """Returns a plain Python list of floats — JSON-serializable, matches
    the 'embeddings as JSON text column' decision from your DB design."""
    model = get_model()
    vec = model.encode([text])[0]
    return vec.tolist()


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
