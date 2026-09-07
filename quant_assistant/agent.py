"""The agent loop: retrieval-augmented generation plus tool calling.

Flow for one question:
  1. Retrieve context from the vector store (RAG) and prepend it to the question.
  2. Send it to Claude with the tool schemas available.
  3. While Claude asks to use tools, execute them and feed the results back.
  4. Return the final grounded answer.

A short rolling history is kept on the instance so follow-up questions in the
same session have lightweight memory of what came before.
"""
from __future__ import annotations

from typing import List

from .config import Config
from .tools import TOOL_SCHEMAS, execute_tool
from .vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are Quant Assistant, a research and analysis assistant for quantitative and "
    "market questions. Ground every answer in the retrieved context and the tools "
    "available to you. When a question needs data exploration, a backtest summary, or a "
    "written report, call the matching tool instead of guessing. Reference the passages "
    "you relied on. If the retrieved context does not support an answer, say so plainly "
    "rather than inventing numbers."
)


class QuantAssistant:
    def __init__(self, store: VectorStore, config: Config | None = None):
        # Imported lazily so retrieval, ingest, and the tests do not require the
        # anthropic package until you actually run the agent.
        from anthropic import Anthropic

        self.store = store
        self.config = config or Config()
        self.client = Anthropic(api_key=self.config.require_api_key())
        self.history: List[dict] = []

    def _retrieved_context(self, question: str) -> str:
        hits = self.store.search(question, k=self.config.top_k)
        if not hits:
            return "(no indexed context found)"
        return "\n\n".join(
            f"[{i + 1}] ({h.source}) {h.text}" for i, h in enumerate(hits)
        )

    def ask(self, question: str, max_steps: int = 6, verbose: bool = False) -> str:
        context = self._retrieved_context(question)
        user_turn = f"Question: {question}\n\nRetrieved context:\n{context}"
        messages = self.history + [{"role": "user", "content": user_turn}]

        for _ in range(max_steps):
            resp = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if b.type == "text")
                self.history = messages
                return answer.strip()

            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    output = execute_tool(block.name, block.input, self.store)
                    if verbose:
                        print(f"  [tool] {block.name}({block.input}) -> {output[:160]}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

        # Step budget exhausted: ask for a final answer without further tools.
        resp = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": "Give your final answer now, based on the tool results so far.",
                }
            ],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
        self.history = messages
        return answer.strip()
