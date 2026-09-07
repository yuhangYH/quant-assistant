"""Minimal FastAPI server for the bilingual Quant Assistant web app.

  uvicorn web.server:app --reload

Loads the prebuilt index at startup (run `ingest` first) and exposes:
  GET  /          the bilingual chat page
  POST /api/ask   {"question": "..."} -> {"answer": "..."}
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quant_assistant.agent import QuantAssistant
from quant_assistant.config import Config
from quant_assistant.vectorstore import VectorStore

app = FastAPI(title="Quant Assistant")

_STATIC = os.path.join(os.path.dirname(__file__), "static")
_config = Config()
_assistant: QuantAssistant | None = None


def get_assistant() -> QuantAssistant:
    global _assistant
    if _assistant is None:
        if not os.path.exists(_config.index_path):
            raise HTTPException(
                status_code=503,
                detail=f"No index at {_config.index_path}. Run `ingest` first.",
            )
        store = VectorStore.load(_config.index_path)
        _assistant = QuantAssistant(store, config=_config)
    return _assistant


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        answer = get_assistant().ask(question)
    except RuntimeError as exc:  # e.g. missing API key
        raise HTTPException(status_code=503, detail=str(exc))
    return AskResponse(answer=answer)
