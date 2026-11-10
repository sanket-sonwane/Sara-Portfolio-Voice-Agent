"""Smallest AI Lightning streaming TTS.

One socket per in-flight segment, drawn from a small pool.

That is the opposite of the Cartesia adapter, which multiplexes many utterances
over one socket, and the difference is protocol, not preference: Lightning echoes
our ``request_id`` as ``external_request_id`` but never echoes ``context_id``, and
two contexts sharing a connection did not each receive their ``complete`` frame.
One request per socket keeps attribution exact and costs little, because a socket
is opened when a segment is submitted and reused by later segments.

Two measured quirks shape this file:

* **Barge-in is a socket close, not a message.** ``context_close: true`` kept
  delivering audio for ~9 s after the call (measured 3 runs, 143-226 late
  chunks). Closing the socket stops it in ~90 ms. A cancelled context therefore
  closes its socket instead of returning it to the pool.
* **``complete`` is not free.** The provider holds it for 4 s after the last
  chunk by default, so every request here asks for ``complete_backoff_ms: 0``.
  Otherwise a turn looks four seconds slower than it is.

Audio is requested as 24 kHz PCM16 and resampled to the 48 kHz egress track with
a stateful resampler; Lightning does not offer 48 kHz, and per-chunk resampling
clicks.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio.frames import StreamResampler
from .base import TTS, AudioSpan, TTSError

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect  # type: ignore

WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"
SAMPLE_RATE = 24_000
# "Rhea" is first in Smallest's recommended Indian female ordering.
DEFAULT_VOICE_ID = "rhea"
DEFAULT_MODEL = "lightning_v3.1_pro"
# "hi" selects Indian-accented English with mid-utterance Hindi code-switching.
# "en" would select UK/US accents, which is the thing we are moving away from.
DEFAULT_LANGUAGE = "hi"


@dataclass(eq=False)
class _Connection:
    """One socket plus the bookkeeping needed to decide whether to reuse it."""

    socket: Any
    last_used: float = field(default_factory=time.monotonic)
    dead: bool = False


class SmallestTTS(TTS):
    name = "smallest"

    def __init__(
        self,
        *,
        api_key: str,
        ws_url: str = WS_URL,
        model: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE_ID,
        language: str = DEFAULT_LANGUAGE,
        sample_rate: int = SAMPLE_RATE,
        open_timeout: float = 6.0,
        response_timeout: float = 20.0,
        max_concurrency: int = 4,
        idle_reuse_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.language = language or DEFAULT_LANGUAGE
        self.sample_rate = sample_rate
        self.open_timeout = open_timeout
        self.response_timeout = response_timeout
        # Lightning allows 10-15 concurrent streams on the pay-as-you-go plan,
        # but the orchestrator fans out one task per answer segment, and a
        # 35-word answer is two or three segments. Four is plenty and keeps a
        # burst from being read as abuse.
        self.max_concurrency = max(1, max_concurrency)
        self.idle_reuse_s = idle_reuse_s
        self._slots = asyncio.Semaphore(self.max_concurrency)
        self._idle: asyncio.Queue[_Connection] = asyncio.Queue()
        self._conns: set[_Connection] = set()
        self._by_context: dict[str, _Connection] = {}

    # ---------------------------------------------------------------- plumbing

    def _url(self) -> str:
        # The provider closes an idle socket after 60 s by default. Our pool
        # discards sockets well before that, but asking for a longer window
        # costs nothing and covers a slow first turn.
        separator = "&" if "?" in self.ws_url else "?"
        return f"{self.ws_url}{separator}timeout=120"

    async def _open(self) -> _Connection:
        try:
            socket = await ws_connect(
                self._url(),
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                max_size=None,
                open_timeout=self.open_timeout,
            )
        except Exception as error:  # noqa: BLE001
            raise TTSError(f"smallest connect failed: {error}") from error
        connection = _Connection(socket)
        self._conns.add(connection)
        return connection

    async def _discard(self, connection: _Connection) -> None:
        # Idempotent: a cancelled segment and the shutdown path can both reach
        # the same connection.
        if connection.dead:
            return
        connection.dead = True
        self._conns.discard(connection)
        try:
            await connection.socket.close()
        except Exception:
            pass

    async def _acquire(self) -> _Connection:
        while True:
            try:
                connection = self._idle.get_nowait()
            except asyncio.QueueEmpty:
                break
            stale = time.monotonic() - connection.last_used > self.idle_reuse_s
            if connection.dead or stale:
                await self._discard(connection)
                continue
            return connection
        return await self._open()

    async def _release(self, connection: _Connection, *, healthy: bool) -> None:
        if not healthy or connection.dead:
            await self._discard(connection)
            return
        connection.last_used = time.monotonic()
        self._idle.put_nowait(connection)

    def _request(self, text: str, context_id: str) -> str:
        return json.dumps(
            {
                "text": text,
                "voice_id": self.voice_id,
                "model": self.model,
                "language": self.language,
                "sample_rate": self.sample_rate,
                "context_id": context_id,
                "request_id": context_id,
                # `continue: false` closes the context and speaks everything
                # buffered. We never use continuations: they buy prosody at the
                # cost of a buffer wait (measured +90-160 ms to first audio).
                "continue": False,
                # Without this the provider waits 4 s after the last chunk
                # before sending `complete`.
                "complete_backoff_ms": 0,
            }
        )

    # -------------------------------------------------------------------- api

    async def prepare(self, text: str) -> None:
        """Open one socket ahead of the first segment, so the greet does not pay it."""
        if self._idle.qsize() or self._conns:
            return
        async with self._slots:
            connection = await self._open()
            self._idle.put_nowait(connection)

    async def synthesize(
        self,
        text: str,
        *,
        context_id: str,
        final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AudioSpan]:
        if not text.strip():
            return
        async with self._slots:
            connection = await self._acquire()
            self._by_context[context_id] = connection
            healthy = False
            try:
                async for span in self._stream(connection, text, context_id):
                    yield span
                healthy = True
            finally:
                self._by_context.pop(context_id, None)
                await self._release(connection, healthy=healthy)

    async def _stream(
        self,
        connection: _Connection,
        text: str,
        context_id: str,
    ) -> AsyncIterator[AudioSpan]:
        resampler = StreamResampler(self.sample_rate, 48_000)
        try:
            await connection.socket.send(self._request(text, context_id))
        except Exception as error:  # noqa: BLE001
            raise TTSError(f"smallest send failed: {error}") from error

        while True:
            try:
                raw = await asyncio.wait_for(
                    connection.socket.recv(), timeout=self.response_timeout
                )
            except TimeoutError as error:
                raise TTSError("smallest timed out waiting for audio") from error
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                raise TTSError(f"smallest receive failed: {error}") from error

            try:
                frame = json.loads(raw)
            except (TypeError, ValueError):
                continue

            status = frame.get("status")
            if status == "error":
                detail = frame.get("message") or frame.get("error") or frame
                raise TTSError(f"smallest error: {detail}")

            data = frame.get("data")
            audio = data.get("audio") if isinstance(data, dict) else None
            if audio:
                samples = np.frombuffer(base64.b64decode(audio), dtype="<i2").astype(
                    np.int16
                )
                converted = resampler.process(samples)
                if converted.size:
                    yield AudioSpan(converted)

            if status == "complete":
                return

    async def cancel(self, context_id: str) -> None:
        """Stop the utterance. Closing the socket is the only thing that works."""
        connection = self._by_context.pop(context_id, None)
        if connection is None:
            return
        await self._discard(connection)

    async def close(self) -> None:
        """Best-effort teardown.

        Shutdown is often already being cancelled by the time this runs, and
        ``websockets.close()`` awaits a shielded waiter that then re-raises
        ``CancelledError``. Nothing useful is left to do at that point, so a
        cancelled teardown must not surface as an error in the lifespan.
        """
        for connection in list(self._conns):
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._discard(connection)
        while not self._idle.empty():
            try:
                self._idle.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def list_voices(self) -> list[dict]:
        import httpx

        url = self.ws_url.replace("wss://", "https://").replace("ws://", "http://")
        base = url.split("/waves/", 1)[0]
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{base}/waves/v1/lightning-v3.1-pro/get_voices",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        return payload.get("voices", payload if isinstance(payload, list) else [])
