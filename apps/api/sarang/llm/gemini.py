"""Gemini streaming adapter (AI Studio key or a Vertex-compatible base URL).

Uses the REST surface directly rather than an SDK: one httpx dependency, no
version skew, and the request shape is visible when something is slow.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import LLM, Done, LLMError, LLMEvent, Message, TextDelta, ToolCall

_GEMINI_TYPES = {
    "object": "OBJECT",
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
}


def _gemini_type(value: str) -> str:
    return _GEMINI_TYPES.get(value.lower(), "STRING")


def _tool_payload(tools: list[dict[str, Any]] | None) -> list[dict] | None:
    if not tools:
        return None
    declarations = []
    for tool in tools:
        function = tool.get("function", {})
        params = function.get("parameters", {}) or {}
        properties = {
            key: {
                "type": _gemini_type(spec.get("type", "string")),
                "description": spec.get("description", ""),
            }
            for key, spec in (params.get("properties") or {}).items()
        }
        declarations.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": params.get("required", []),
                },
            }
        )
    return [{"functionDeclarations": declarations}]


def _contents(messages: list[Message]) -> tuple[dict | None, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        if message.role == "system":
            system_parts.append(message.content)
        elif message.role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": message.name or "search_portfolio",
                                "response": {"result": message.content},
                            }
                        }
                    ],
                }
            )
        elif message.role == "assistant":
            contents.append({"role": "model", "parts": [{"text": message.content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": message.content}]})
    system = {"parts": [{"text": "\n".join(system_parts)}]} if system_parts else None
    return system, contents


class GeminiLLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 30.0,
        name: str = "gemini",
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0))
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 160,
        temperature: float = 0.3,
    ) -> AsyncIterator[LLMEvent]:
        system, contents = _contents(messages)
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = system
        tool_payload = _tool_payload(tools)
        if tool_payload:
            payload["tools"] = tool_payload

        url = f"{self.base_url}/models/{self.model}:streamGenerateContent"
        params = {"alt": "sse", "key": self.api_key}
        usage: dict[str, Any] = {}
        finish_reason = "stop"

        try:
            client = await self._get_client()
            async with client.stream("POST", url, params=params, json=payload) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "ignore")[:400]
                        raise LLMError(f"gemini HTTP {response.status_code}: {body}")
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            continue
                        if event.get("usageMetadata"):
                            usage = event["usageMetadata"]
                        for candidate in event.get("candidates") or []:
                            if candidate.get("finishReason"):
                                finish_reason = str(candidate["finishReason"]).lower()
                            for part in (candidate.get("content") or {}).get("parts") or []:
                                if part.get("text"):
                                    yield TextDelta(part["text"])
                                call = part.get("functionCall")
                                if call and call.get("name"):
                                    yield ToolCall(
                                        id=call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                        name=call["name"],
                                        arguments=call.get("args") or {},
                                    )
        except httpx.HTTPError as error:
            raise LLMError(f"gemini transport error: {error}") from error

        yield Done(usage=usage, finish_reason=finish_reason)
