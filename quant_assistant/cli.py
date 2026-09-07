"""Command-line entry point.

  python -m quant_assistant.cli ingest --notes data/sample/research_notes.md --data data/sample/prices.csv
  python -m quant_assistant.cli ask "How volatile has ACME been, and what do the notes say about momentum?"
"""
from __future__ import annotations

import argparse
import os
import sys

from .agent import QuantAssistant
from .config import Config
from .ingest import build_index
from .vectorstore import VectorStore


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = Config()
    _, n = build_index(args.notes, args.data, config=config)
    print(f"Indexed {n} passages -> {config.index_path} (embeddings: {config.embeddings_backend})")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    config = Config()
    if not os.path.exists(config.index_path):
        print(f"No index at {config.index_path}. Run `ingest` first.", file=sys.stderr)
        return 1
    store = VectorStore.load(config.index_path)
    assistant = QuantAssistant(store, config=config)
    answer = assistant.ask(args.question, verbose=args.verbose)
    print(answer)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="quant-assistant", description="Quant Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build the retrieval index.")
    p_ingest.add_argument("--notes", default="data/sample/research_notes.md")
    p_ingest.add_argument("--data", default="data/sample/prices.csv")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question.")
    p_ask.add_argument("question")
    p_ask.add_argument("--verbose", action="store_true", help="Print tool calls as they happen.")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
