"""LLM adapter contract.

Providers differ in transport, not in shape: every adapter yields text deltas and,
optionally, tool calls. The orchestrator owns the tool loop, so adding a provider
never touches the turn engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    name: str = ""
    tool_call_id: str = ""

    def to_openai(self) -> dict[str, Any]:
        base: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool":
            base["tool_call_id"] = self.tool_call_id
            base["name"] = self.name
        return base


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Done:
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"


LLMEvent = TextDelta | ToolCall | Done

TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_portfolio",
            "description": (
                "Search the long-form portfolio evidence: case studies, research notes "
                "and engineering details. Use only when the supplied facts are not "
                "enough to answer with specifics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                },
                "required": ["query"],
            },
        },
    }
]


class LLM(ABC):
    name: str = "llm"

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 160,
        temperature: float = 0.3,
    ) -> AsyncIterator[LLMEvent]:
        """Yield deltas. Closing the iterator must close the upstream request."""
        raise NotImplementedError

    async def close(self) -> None:
        return None


class LLMError(RuntimeError):
    """A provider failed. The orchestrator falls back to the local answerer."""
