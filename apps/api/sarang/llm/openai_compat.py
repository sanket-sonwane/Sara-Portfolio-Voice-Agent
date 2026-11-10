"""OpenAI-compatible chat streaming (Groq, Sarvam).

One implementation, two deployments: the only differences are the base URL, the
auth header and the model id. Both speak SSE and both support tool calls.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import LLM, Done, LLMError, LLMEvent, Message, TextDelta, ToolCall


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer ",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            self.auth_header: f"{self.auth_prefix}{self.api_key}",
        }
        headers.update(self.extra_headers)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                headers=self._headers(),
            )
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.extra_body:
            payload.update(self.extra_body)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        tool_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason = "stop"

        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "ignore")[:400]
                        raise LLMError(f"{self.name} HTTP {response.status_code}: {body}")
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            continue
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            for call in delta.get("tool_calls") or []:
                                index = int(call.get("index", 0))
                                slot = tool_acc.setdefault(
                                    index,
                                    {"id": call.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                     "name": "", "arguments": ""},
                                )
                                if call.get("function", {}).get("name"):
                                    slot["name"] = call["function"]["name"]
                                if call.get("function", {}).get("arguments"):
                                    slot["arguments"] += call["function"]["arguments"]
                            content = delta.get("content")
                            if content:
                                yield TextDelta(content)
        except httpx.HTTPError as error:
            raise LLMError(f"{self.name} transport error: {error}") from error

        for slot in tool_acc.values():
            if not slot["name"]:
                continue
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except ValueError:
                arguments = {"query": slot["arguments"]}
            yield ToolCall(id=slot["id"], name=slot["name"], arguments=arguments)

        yield Done(usage=usage, finish_reason=finish_reason)
