"""Pluggable text embeddings.

Three backends, resolved in this order when EMBEDDINGS_BACKEND=auto:

  1. voyage               Voyage AI (Anthropic's recommended embedding provider),
                          used when VOYAGE_API_KEY is set and the voyageai package
                          is installed. This is the intended production path.
  2. sentence-transformers  A local transformer model (all-MiniLM-L6-v2). Real
                          dense embeddings, no API key, runs offline after the
                          first model download.
  3. hashing              A dependency-free hashed bag-of-words vectorizer. Not a
                          learned embedding, but a real, deterministic vector space
                          that keeps retrieval working (and the tests fast) anywhere.

You can force a backend with EMBEDDINGS_BACKEND=voyage|sentence-transformers|hashing.
"""
from __future__ import annotations

import os
import re
import zlib
from typing import List

import numpy as np

_HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lazily created singletons so we only pay init cost once.
_st_model = None
_voyage_client = None


def resolve_backend(preference: str = "auto") -> str:
    if preference and preference != "auto":
        return preference
    if os.getenv("VOYAGE_API_KEY"):
        try:
            import voyageai  # noqa: F401
            return "voyage"
        except ImportError:
            pass
    try:
        import sentence_transformers  # noqa: F401
        return "sentence-transformers"
    except ImportError:
        return "hashing"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _hash_embed(texts: List[str]) -> np.ndarray:
    vecs = np.zeros((len(texts), _HASH_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in _TOKEN_RE.findall(text.lower()):
            # zlib.crc32 is deterministic across processes, unlike the salted built-in hash().
            vecs[i, zlib.crc32(tok.encode("utf-8")) % _HASH_DIM] += 1.0
    return _l2_normalize(vecs)


def _voyage_embed(texts: List[str], input_type: str) -> np.ndarray:
    global _voyage_client
    import voyageai

    if _voyage_client is None:
        _voyage_client = voyageai.Client()
    model = os.getenv("VOYAGE_MODEL", "voyage-3")
    result = _voyage_client.embed(texts, model=model, input_type=input_type)
    return _l2_normalize(np.array(result.embeddings, dtype=np.float32))


def _st_embed(texts: List[str]) -> np.ndarray:
    global _st_model
    from sentence_transformers import SentenceTransformer

    if _st_model is None:
        name = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
        _st_model = SentenceTransformer(name)
    vecs = _st_model.encode(texts, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_texts(texts: List[str], backend: str = "auto") -> np.ndarray:
    """Embed a list of documents. Returns an (n, d) float32 matrix."""
    backend = resolve_backend(backend)
    if backend == "voyage":
        return _voyage_embed(texts, input_type="document")
    if backend == "sentence-transformers":
        return _st_embed(texts)
    return _hash_embed(texts)


def embed_query(text: str, backend: str = "auto") -> np.ndarray:
    """Embed a single query. Returns a (d,) float32 vector."""
    backend = resolve_backend(backend)
    if backend == "voyage":
        return _voyage_embed([text], input_type="query")[0]
    if backend == "sentence-transformers":
        return _st_embed([text])[0]
    return _hash_embed([text])[0]
