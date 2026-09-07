"""Runtime configuration for Quant Assistant.

Everything is driven by environment variables so the same code runs from the
CLI, the web server, and the tests without edits. Copy .env.example to .env and
fill in your keys, or export the variables directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Config:
    # Claude
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("QA_MODEL", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: int(_env("QA_MAX_TOKENS", "1500")))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("QA_TOP_K", "5")))
    embeddings_backend: str = field(default_factory=lambda: _env("EMBEDDINGS_BACKEND", "auto"))

    # Storage
    index_path: str = field(default_factory=lambda: _env("QA_INDEX", "index.pkl"))

    def require_api_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export ANTHROPIC_API_KEY before running."
            )
        return self.anthropic_api_key
