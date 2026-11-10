"""Tool execution.

One tool in v1: ``search_portfolio`` over the long-form evidence in ``kb/docs``.
It is only reached when tier-1 lookup says the question needs depth, which keeps
the common turn to zero retrieval calls.

The runner is deliberately strict:

* bounded wall time — a slow retrieval must not hold the turn hostage;
* bounded output — the tool result is read aloud and paid for in prompt tokens;
* no writes, no network, no arbitrary code.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .kb.retrieval import get_retriever, search_docs
from .llm.base import TOOL_SCHEMA


@dataclass
class ToolResult:
    name: str
    call_id: str
    ok: bool
    content: str
    duration_ms: float

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "call_id": self.call_id,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 1),
            "chars": len(self.content),
        }


class ToolRunner:
    def __init__(self, *, timeout_s: float = 0.9, max_chars: int = 900) -> None:
        self.timeout_s = timeout_s
        self.max_chars = max_chars
        self.calls = 0
        self.timeouts = 0

    @property
    def ready(self) -> bool:
        return get_retriever().ready

    async def execute(self, name: str, call_id: str, arguments: dict) -> ToolResult:
        started = time.perf_counter()
        if name != "search_portfolio":
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                content=f"Unknown tool {name!r}. Answer from the supplied facts only.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(
                name=name, call_id=call_id, ok=False, content="No search query was supplied.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        self.calls += 1
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(search_docs, query, 3), timeout=self.timeout_s
            )
        except TimeoutError:
            self.timeouts += 1
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                content=(
                    "The case-study search timed out. Answer from the supplied facts, "
                    "or say the detail is not documented."
                ),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as error:  # noqa: BLE001
            return ToolResult(
                name=name,
                call_id=call_id,
                ok=False,
                content=f"Search failed ({type(error).__name__}). Use the supplied facts only.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        if not chunks:
            content = "No case-study evidence matched that query."
        else:
            sections: list[str] = []
            used = 0
            for chunk in chunks:
                block = f"[{chunk.doc_id}] {chunk.title}: {chunk.text}"
                if used + len(block) > self.max_chars:
                    if not sections:
                        # Always return the best chunk, truncated, rather than
                        # an empty result that reads as "nothing exists".
                        sections.append(block[: self.max_chars])
                    break
                sections.append(block)
                used += len(block) + 1
            content = "\n".join(sections) or "No usable case-study evidence."

        return ToolResult(
            name=name,
            call_id=call_id,
            ok=True,
            content=content,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def tool_schema() -> list[dict]:
    return TOOL_SCHEMA
