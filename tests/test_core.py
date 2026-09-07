"""Offline tests for the parts that do not need the Claude API.

They force the dependency-free hashing embedding backend so they run anywhere,
with no API key and no model download.
"""
import json
import os

import pandas as pd

from quant_assistant.tools import execute_tool
from quant_assistant.vectorstore import VectorStore

os.environ.setdefault("EMBEDDINGS_BACKEND", "hashing")


def _store():
    store = VectorStore(backend="hashing")
    store.add(
        [
            "Momentum dominates at the monthly horizon while mean reversion lives intramonth.",
            "Realized volatility clusters, with drawdowns arriving after the strongest up months.",
            "Volume rises on up months and thins on pullbacks, which affects execution.",
        ],
        ["notes", "notes", "notes"],
    )
    store.attach_frame(
        pd.DataFrame({"close": [100.0, 103.5, 101.2, 105.8], "ret": [0.0, 0.035, -0.022, 0.045]})
    )
    return store


def test_retrieval_ranks_relevant_passage_first():
    store = _store()
    hits = store.search("Is there a momentum or mean reversion effect?", k=3)
    assert hits, "expected at least one hit"
    assert "momentum" in hits[0].text.lower()


def test_explore_features_reports_stats():
    store = _store()
    out = json.loads(execute_tool("explore_features", {"column": "close"}, store))
    assert out["count"] == 4
    assert out["latest"] == 105.8
    assert out["max"] == 105.8


def test_explore_features_unknown_column():
    store = _store()
    out = execute_tool("explore_features", {"column": "nope"}, store)
    assert "not found" in out


def test_summarize_backtest_math():
    out = json.loads(
        execute_tool(
            "summarize_backtest",
            {"returns": [0.02, -0.01, 0.03, 0.00, -0.02, 0.04], "periods_per_year": 12},
            store=None,
        )
    )
    assert out["n_periods"] == 6
    # cumulative = prod(1+r) - 1
    assert abs(out["cumulative_return"] - 0.060064) < 1e-4
    assert out["max_drawdown"] <= 0.0


def test_generate_report_formats_markdown():
    out = execute_tool(
        "generate_report", {"title": "ACME", "findings": ["Up trend", "Vol clusters"]}, store=None
    )
    assert out.startswith("## ACME")
    assert "- Up trend" in out


def test_index_roundtrip(tmp_path):
    store = _store()
    path = str(tmp_path / "index.pkl")
    store.save(path)
    loaded = VectorStore.load(path)
    hits = loaded.search("volatility clustering", k=1)
    assert hits and "volatility" in hits[0].text.lower()
