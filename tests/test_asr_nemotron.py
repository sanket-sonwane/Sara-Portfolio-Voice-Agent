"""Nemotron ASR lifecycles: batch finals and streaming turn tagging."""

from __future__ import annotations

import asyncio

import numpy as np
from sarang.asr import nemotron
from sarang.asr.base import ASREvent
from sarang.asr.nemotron import NemotronASR, NemotronStreamASR


def pcm(ms: int = 100) -> bytes:
    return np.zeros(16 * ms, dtype=np.int16).tobytes()


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"text": "hello from the batch recogniser"}


class _FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def post(self, *args, **kwargs):
        return _FakeResponse()


async def _next_event(asr: NemotronASR) -> ASREvent:
    async for event in asr.events():
        return event
    raise AssertionError("event stream ended without an event")


async def test_batch_event_reader_survives_until_finalize(monkeypatch):
    """The reader starts before any transcript exists; it must keep waiting."""
    monkeypatch.setattr(nemotron.httpx, "AsyncClient", _FakeClient)
    asr = NemotronASR(api_key="test", url="https://example.test/stt", model="nemotron")

    reader = asyncio.create_task(_next_event(asr))
    await asyncio.sleep(0.01)
    assert not reader.done(), "the event reader exited before finalize"

    await asr.push(pcm())
    await asr.finalize()

    event = await asyncio.wait_for(reader, timeout=1.0)
    assert event.text == "hello from the batch recogniser"
    assert event.final is True


async def test_batch_close_ends_the_event_stream():
    asr = NemotronASR(api_key="test", url="https://example.test/stt", model="nemotron")
    stream = asr.events()
    pending = asyncio.create_task(stream.__anext__())

    await asr.close()

    try:
        await asyncio.wait_for(pending, timeout=1.0)
    except StopAsyncIteration:
        pass
    else:  # pragma: no cover - the generator must end
        raise AssertionError("closing the adapter did not end events()")


async def test_stream_reports_dead_until_connected():
    asr = NemotronStreamASR(url="wss://example.test/stream")

    assert asr.alive is False
    assert await asr.begin_utterance() is False


async def test_stream_tags_each_turn_instead_of_reusing_zero():
    asr = NemotronStreamASR(url="wss://example.test/stream")
    # Stand in for a live connection without opening a socket.
    asr._socket = object()
    asr._reader = asyncio.create_task(asyncio.sleep(5))
    try:
        assert asr.alive is True
        assert await asr.begin_utterance() is True
        assert asr._turn == 1
        assert await asr.begin_utterance() is True
        assert asr._turn == 2
    finally:
        asr._reader.cancel()
        await asyncio.gather(asr._reader, return_exceptions=True)
        asr._reader = None

    assert asr.alive is False
