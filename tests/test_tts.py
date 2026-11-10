"""TTS adapters: concurrency, attribution and cancellation.

Cartesia multiplexes many utterances over one socket, so the bug these pin is a
lock around the whole receive loop that made a second speaker wait for the first
one's audio to finish. Sessions must not serialise.

Smallest Lightning is the opposite protocol: it does not echo ``context_id`` and
does not reliably complete two contexts on one socket, so its tests pin the
inverse property - one socket per in-flight segment, reused sequentially.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from sarang.tts import cartesia as cartesia_module
from sarang.tts import smallest as smallest_module
from sarang.tts.cartesia import CartesiaTTS
from sarang.tts.smallest import SmallestTTS


def _frame(samples: list[int]) -> str:
    pcm = np.array(samples, dtype="<i2").tobytes()
    return json.dumps({"context_id": None, "data": base64.b64encode(pcm).decode()})


class FakeSocket:
    """Minimal Cartesia socket: each request answers after its own delay."""

    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False
        self.in_flight = 0
        self.max_in_flight = 0

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        if payload.get("cancel"):
            return
        self.sent.append(payload)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        context_id = payload["context_id"]
        asyncio.create_task(self._answer(context_id, self.delays.get(context_id, 0.0)))

    async def _answer(self, context_id: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if self.closed:
            return
        pcm = np.array([100, 200, 300, 400] * 120, dtype="<i2").tobytes()
        encoded = base64.b64encode(pcm).decode()
        # Two chunks: the stateful resampler buffers the first one.
        await self.incoming.put(json.dumps({"context_id": context_id, "data": encoded}))
        await self.incoming.put(json.dumps({"context_id": context_id, "data": encoded}))
        await self.incoming.put(json.dumps({"context_id": context_id, "done": True}))
        self.in_flight -= 1

    async def recv(self) -> str:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True


async def _collect(tts: CartesiaTTS, text: str, context_id: str) -> int:
    total = 0
    async for span in tts.synthesize(text, context_id=context_id):
        total += int(span.pcm.size)
    return total


async def test_two_utterances_run_concurrently(monkeypatch):
    socket = FakeSocket({"slow": 0.2, "fast": 0.0})

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(cartesia_module, "ws_connect", fake_connect)
    tts = CartesiaTTS(api_key="test", ws_url="wss://example.test/tts/ws")

    slow = asyncio.create_task(_collect(tts, "one", "slow"))
    await asyncio.sleep(0.02)
    fast = asyncio.create_task(_collect(tts, "two", "fast"))

    fast_samples = await asyncio.wait_for(fast, timeout=0.15)
    assert fast_samples > 0
    assert not slow.done(), "the fast utterance must not wait for the slow one"

    slow_samples = await asyncio.wait_for(slow, timeout=1.0)
    assert slow_samples > 0
    assert {payload["context_id"] for payload in socket.sent} == {"slow", "fast"}


async def test_concurrent_contexts_are_capped(monkeypatch):
    # The account allows two concurrent contexts. The orchestrator submits every
    # answer segment at once, so without a bound a three-sentence answer tripped
    # "concurrency limit of 2" and the synthetic voice leaked into the reply.
    socket = FakeSocket({f"c{index}": 0.05 for index in range(5)})

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(cartesia_module, "ws_connect", fake_connect)
    tts = CartesiaTTS(
        api_key="test", ws_url="wss://example.test/tts/ws", max_concurrency=2
    )

    results = await asyncio.gather(
        *(_collect(tts, "text", f"c{index}") for index in range(5))
    )

    assert all(samples > 0 for samples in results)
    assert socket.max_in_flight <= 2


async def test_audio_is_routed_by_context_id(monkeypatch):
    socket = FakeSocket({"a": 0.0, "b": 0.0})

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(cartesia_module, "ws_connect", fake_connect)
    tts = CartesiaTTS(api_key="test", ws_url="wss://example.test/tts/ws")

    # A stray frame for an unknown context must be ignored, not delivered.
    await socket.incoming.put(_frame([9, 9, 9]))
    both = await asyncio.gather(_collect(tts, "a", "a"), _collect(tts, "b", "b"))
    assert all(samples > 0 for samples in both)


async def test_cancel_releases_the_waiter(monkeypatch):
    socket = FakeSocket({"ctx": 5.0})

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(cartesia_module, "ws_connect", fake_connect)
    tts = CartesiaTTS(api_key="test", ws_url="wss://example.test/tts/ws")

    task = asyncio.create_task(_collect(tts, "hello", "ctx"))
    await asyncio.sleep(0.02)
    await tts.cancel("ctx")
    # Cancelling ends the utterance cleanly; it is not an error.
    assert await asyncio.wait_for(task, timeout=1.0) == 0


async def test_provider_error_is_raised_for_the_matching_context(monkeypatch):
    socket = FakeSocket({"ctx": 5.0})

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(cartesia_module, "ws_connect", fake_connect)
    tts = CartesiaTTS(api_key="test", ws_url="wss://example.test/tts/ws")

    task = asyncio.create_task(_collect(tts, "hello", "ctx"))
    await asyncio.sleep(0.02)
    await socket.incoming.put(
        json.dumps({"context_id": "ctx", "type": "error", "error": "boom"})
    )
    with pytest.raises(cartesia_module.TTSError):
        await asyncio.wait_for(task, timeout=1.0)


# --------------------------------------------------------------------- smallest


class FakeSmallestSocket:
    """One Lightning socket: answers each request, then reports `complete`."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False
        self.in_flight = 0
        self.max_in_flight = 0

    async def send(self, raw: str) -> None:
        payload = json.loads(raw)
        self.sent.append(payload)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        asyncio.create_task(self._answer(self.delay))

    async def _answer(self, delay: float) -> None:
        await asyncio.sleep(delay)
        if self.closed:
            return
        pcm = np.array([100, 200, 300, 400] * 120, dtype="<i2").tobytes()
        encoded = base64.b64encode(pcm).decode()
        # Three chunks: the stateful resampler buffers the first one.
        for _ in range(3):
            await self.incoming.put(
                json.dumps({"status": "chunk", "data": {"audio": encoded}})
            )
        await self.incoming.put(json.dumps({"status": "complete"}))
        self.in_flight -= 1

    async def recv(self) -> str:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True


async def _drain(tts) -> int:
    total = 0
    async for span in tts.synthesize("text", context_id="ctx"):
        total += int(span.pcm.size)
    return total


async def test_smallest_opens_one_socket_per_segment(monkeypatch):
    """Two segments in flight must not share a connection.

    Lightning does not echo `context_id`, and two contexts on one socket did not
    each get a `complete`. Sharing would therefore mis-attribute audio.
    """
    sockets: list[FakeSmallestSocket] = []

    async def fake_connect(*args, **kwargs):
        socket = FakeSmallestSocket(delay=0.05)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")

    results = await asyncio.gather(
        tts.synthesize("one", context_id="a").__anext__(),
        tts.synthesize("two", context_id="b").__anext__(),
    )

    assert all(span.pcm.size > 0 for span in results)
    assert len(sockets) == 2
    assert all(socket.max_in_flight == 1 for socket in sockets)


async def test_smallest_reuses_a_socket_sequentially(monkeypatch):
    sockets: list[FakeSmallestSocket] = []

    async def fake_connect(*args, **kwargs):
        socket = FakeSmallestSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")

    first = await _drain(tts)
    second = await _drain(tts)

    assert first > 0 and second > 0
    assert len(sockets) == 1


async def test_smallest_request_shape(monkeypatch):
    sockets: list[FakeSmallestSocket] = []

    async def fake_connect(*args, **kwargs):
        socket = FakeSmallestSocket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live", language="hi")

    await _drain(tts)

    payload = sockets[0].sent[0]
    assert payload["complete_backoff_ms"] == 0, "the 4 s default would inflate every turn"
    assert payload["continue"] is False
    assert payload["language"] == "hi"
    assert payload["voice_id"] == "rhea"
    assert payload["model"] == "lightning_v3.1_pro"


async def test_smallest_cancel_closes_the_socket(monkeypatch):
    """`context_close` kept sending audio for ~9 s; closing the socket is the fix."""
    sockets: list[FakeSmallestSocket] = []

    async def fake_connect(*args, **kwargs):
        socket = FakeSmallestSocket(delay=5.0)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")

    task = asyncio.create_task(_drain(tts))
    await asyncio.sleep(0.02)
    await tts.cancel("ctx")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sockets[0].closed is True


async def test_smallest_error_frame_raises(monkeypatch):
    socket = FakeSmallestSocket(delay=5.0)

    async def fake_connect(*args, **kwargs):
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")

    task = asyncio.create_task(_drain(tts))
    await asyncio.sleep(0.02)
    await socket.incoming.put(json.dumps({"status": "error", "message": "boom"}))
    with pytest.raises(smallest_module.TTSError):
        await asyncio.wait_for(task, timeout=1.0)


async def test_smallest_cancel_twice_is_a_no_op(monkeypatch):
    """Cancel and the cancelled segment's cleanup both reach the connection."""
    sockets: list[FakeSmallestSocket] = []

    async def fake_connect(*args, **kwargs):
        socket = FakeSmallestSocket(delay=5.0)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")

    task = asyncio.create_task(_drain(tts))
    await asyncio.sleep(0.02)
    await tts.cancel("ctx")
    await tts.cancel("ctx")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sockets[0].closed is True


async def test_smallest_close_survives_a_cancelled_teardown(monkeypatch):
    """Lifespan shutdown is often already cancelled; that must not become an error."""

    class CancelledCloseSocket(FakeSmallestSocket):
        async def close(self) -> None:
            self.closed = True
            raise asyncio.CancelledError

    async def fake_connect(*args, **kwargs):
        return CancelledCloseSocket()

    monkeypatch.setattr(smallest_module, "ws_connect", fake_connect)
    tts = SmallestTTS(api_key="test", ws_url="wss://example.test/live")
    await tts.prepare("hello")

    await tts.close()
