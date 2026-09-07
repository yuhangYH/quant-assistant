"""A small in-memory vector store with cosine similarity search.

It holds two things:
  * an embedded text corpus (research notes and a description of the dataset),
    used for retrieval-augmented generation;
  * the raw structured table (a pandas DataFrame), used by the analysis tools.

Both are persisted together so `ingest` runs once and `ask` / the web server
load a ready index.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .embeddings import embed_query, embed_texts


@dataclass
class Hit:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.frame: Optional[pd.DataFrame] = None

    def add(self, texts: List[str], sources: List[str]) -> None:
        if len(texts) != len(sources):
            raise ValueError("texts and sources must be the same length")
        new_vecs = embed_texts(texts, backend=self.backend)
        self.texts.extend(texts)
        self.sources.extend(sources)
        self.vectors = new_vecs if self.vectors is None else np.vstack([self.vectors, new_vecs])

    def attach_frame(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def search(self, query: str, k: int = 5) -> List[Hit]:
        if self.vectors is None or not self.texts:
            return []
        q = embed_query(query, backend=self.backend)
        scores = self.vectors @ q  # vectors are L2-normalized, so this is cosine similarity
        order = np.argsort(-scores)[: max(1, k)]
        return [Hit(self.texts[i], self.sources[i], float(scores[i])) for i in order]

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "backend": self.backend,
                    "texts": self.texts,
                    "sources": self.sources,
                    "vectors": self.vectors,
                    "frame": self.frame,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        store = cls(backend=state.get("backend", "auto"))
        store.texts = state["texts"]
        store.sources = state["sources"]
        store.vectors = state["vectors"]
        store.frame = state["frame"]
        return store
