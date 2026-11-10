"""Tool runner: bounded time, bounded output, no writes."""

from __future__ import annotations

import time

import sarang.tools as tools_module
from sarang.kb.retrieval import Chunk
from sarang.tools import ToolRunner


async def test_unknown_tool_is_refused():
    result = await ToolRunner().execute("delete_everything", "call_1", {})
    assert not result.ok
    assert "Unknown tool" in result.content


async def test_empty_query_is_refused():
    result = await ToolRunner().execute("search_portfolio", "call_2", {"query": ""})
    assert not result.ok
    assert "No search query" in result.content


async def test_happy_path_returns_bounded_evidence():
    result = await ToolRunner(max_chars=200).execute(
        "search_portfolio", "call_3", {"query": "signed spend approval"}
    )
    assert result.ok
    assert 0 < len(result.content) <= 200 + 200
    assert "agentic-commerce-gateway" in result.content


async def test_timeout_is_reported_not_raised(monkeypatch):
    def slow_search(query, k=3):
        time.sleep(0.3)
        return [Chunk("x", "x", "x")]

    monkeypatch.setattr(tools_module, "search_docs", slow_search)
    runner = ToolRunner(timeout_s=0.05)
    result = await runner.execute("search_portfolio", "call_4", {"query": "anything"})
    assert not result.ok
    assert "timed out" in result.content.lower()
    assert runner.timeouts == 1


async def test_no_matches_is_an_answer_not_an_error(monkeypatch):
    monkeypatch.setattr(tools_module, "search_docs", lambda query, k=3: [])
    result = await ToolRunner().execute(
        "search_portfolio", "call_5", {"query": "nonexistent topic xyzzy"}
    )
    assert result.ok
    assert "No case-study evidence" in result.content


async def test_payload_is_small_and_omits_the_body():
    result = await ToolRunner().execute(
        "search_portfolio", "call_6", {"query": "signed spend approval"}
    )
    payload = result.to_payload()
    assert set(payload) == {"name", "call_id", "ok", "duration_ms", "chars"}
    assert payload["chars"] > 0
