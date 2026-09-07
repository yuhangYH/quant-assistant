"""Quant Assistant: a small, real RAG + tool-calling research assistant built on Claude."""

from .agent import QuantAssistant
from .config import Config
from .vectorstore import VectorStore

__all__ = ["QuantAssistant", "Config", "VectorStore"]
__version__ = "0.1.0"
