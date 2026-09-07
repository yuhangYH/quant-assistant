"""Build the retrieval index from a research-notes file and a structured CSV.

Chunking here is intentionally simple and readable: split the markdown notes on
blank lines, keep chunks with enough substance, and add one synthetic passage
that describes the structured dataset so a question about the data can retrieve
its shape before the tools are called.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd

from .config import Config
from .vectorstore import VectorStore

_MIN_CHUNK_CHARS = 40


def chunk_notes(text: str) -> List[str]:
    raw = [block.strip() for block in text.split("\n\n")]
    return [block for block in raw if len(block) >= _MIN_CHUNK_CHARS]


def _describe_frame(frame: pd.DataFrame, name: str) -> str:
    cols = ", ".join(map(str, frame.columns))
    return (
        f"Structured dataset '{name}' has {len(frame)} rows and columns: {cols}. "
        f"Use the explore_features tool to compute statistics on any numeric column."
    )


def build_index(
    notes_path: str,
    csv_path: str,
    config: Config | None = None,
    dataset_name: str = "prices",
) -> Tuple[VectorStore, int]:
    config = config or Config()
    store = VectorStore(backend=config.embeddings_backend)

    texts: List[str] = []
    sources: List[str] = []

    if os.path.exists(notes_path):
        with open(notes_path, "r", encoding="utf-8") as fh:
            for chunk in chunk_notes(fh.read()):
                texts.append(chunk)
                sources.append(os.path.basename(notes_path))

    frame = None
    if os.path.exists(csv_path):
        frame = pd.read_csv(csv_path)
        texts.append(_describe_frame(frame, dataset_name))
        sources.append(os.path.basename(csv_path))

    if texts:
        store.add(texts, sources)
    if frame is not None:
        store.attach_frame(frame)

    store.save(config.index_path)
    return store, len(texts)
