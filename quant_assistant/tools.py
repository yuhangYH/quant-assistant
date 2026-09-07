"""Tools the agent can call.

Each tool has a JSON schema (advertised to Claude) and a Python implementation.
The tools are deliberately small and deterministic so their output is easy to
verify and easy to reason about during an interview walkthrough.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "retrieve",
        "description": (
            "Semantic search over the indexed corpus of research notes and the "
            "dataset description. Use it to pull supporting context before answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "k": {"type": "integer", "description": "How many passages to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "explore_features",
        "description": (
            "Compute summary statistics (count, mean, std, min, max, latest) for one "
            "numeric column of the structured dataset. Use it before making any claim "
            "about the data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Column name in the dataset."},
            },
            "required": ["column"],
        },
    },
    {
        "name": "summarize_backtest",
        "description": (
            "Given a series of periodic returns (as decimals, e.g. 0.01 for 1%), compute "
            "cumulative return, annualized volatility, Sharpe ratio, and maximum drawdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "returns": {"type": "array", "items": {"type": "number"}},
                "periods_per_year": {
                    "type": "integer",
                    "description": "Annualization factor, e.g. 252 for daily, 12 for monthly (default 252).",
                },
            },
            "required": ["returns"],
        },
    },
    {
        "name": "generate_report",
        "description": "Format a short, clean markdown report from a title and a list of findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "findings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "findings"],
        },
    },
]


def _retrieve(store, query: str, k: int = 5) -> str:
    hits = store.search(query, k=k)
    if not hits:
        return "No indexed context found. The index may be empty; run `ingest` first."
    lines = [f"[{i + 1}] ({h.source}, score={h.score:.3f}) {h.text}" for i, h in enumerate(hits)]
    return "\n".join(lines)


def _explore_features(store, column: str) -> str:
    if store.frame is None:
        return "No structured dataset is loaded."
    if column not in store.frame.columns:
        cols = ", ".join(map(str, store.frame.columns))
        return f"Column '{column}' not found. Available columns: {cols}."
    series = store.frame[column]
    numeric = series.apply(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)).all()
    if not numeric:
        return f"Column '{column}' is not numeric; summary statistics do not apply."
    values = series.astype(float)
    stats = {
        "column": column,
        "count": int(values.count()),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "latest": round(float(values.iloc[-1]), 6),
    }
    return json.dumps(stats)


def _summarize_backtest(returns: List[float], periods_per_year: int = 252) -> str:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return "Empty return series."
    curve = np.cumprod(1.0 + r)
    cumulative = float(curve[-1] - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0
    mean_annual = float(r.mean() * periods_per_year)
    sharpe = round(mean_annual / vol, 4) if vol > 0 else None
    running_max = np.maximum.accumulate(curve)
    max_drawdown = float((curve / running_max - 1.0).min())
    result = {
        "n_periods": int(r.size),
        "cumulative_return": round(cumulative, 6),
        "annualized_volatility": round(vol, 6),
        "sharpe_ratio": sharpe,
        "max_drawdown": round(max_drawdown, 6),
    }
    return json.dumps(result)


def _generate_report(title: str, findings: List[str]) -> str:
    body = "\n".join(f"- {f}" for f in findings)
    return f"## {title}\n\n{body}\n"


def execute_tool(name: str, tool_input: Dict[str, Any], store) -> str:
    """Dispatch a tool call by name and return its result as a string."""
    if name == "retrieve":
        return _retrieve(store, tool_input["query"], int(tool_input.get("k", 5)))
    if name == "explore_features":
        return _explore_features(store, tool_input["column"])
    if name == "summarize_backtest":
        return _summarize_backtest(
            tool_input["returns"], int(tool_input.get("periods_per_year", 252))
        )
    if name == "generate_report":
        return _generate_report(tool_input["title"], tool_input.get("findings", []))
    return f"Unknown tool: {name}"
