"""Cartesia Sonic streaming TTS.

One WebSocket per session, many contexts. ``context_id`` is derived from the turn
generation, so a barge-in can cancel the exact utterance without guessing which
sockets are still open.

The socket has a single reader task that demultiplexes audio by ``context_id``
into per-utterance queues. A lock around the whole receive loop would serialise
every concurrent speaker behind the first one; this keeps turns independent.

Audio is requested as raw PCM16 at 24 kHz and resampled to the 48 kHz track with a
stateful resampler: per-chunk resampling clicks, and clicks read as latency.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from ..audio.frames import StreamResampler
from .base import TTS, AudioSpan, TTSError

try:  # websockets >= 13
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover
    from websockets import connect as ws_connect  # type: ignore

API_VERSION = "2024-06-10"
SAMPLE_RATE = 24_000
# "Ella - Caring Scout": feminine, en-AU. Overridden by CARTESIA_VOICE_ID.
# Cartesia's catalogue is not India-accented; that is the trade-off for its much
# lower TTFA than Bulbul.
DEFAULT_VOICE_ID = "2a12b36c-7f9b-4c3a-9f7a-72731b15323a"

_END = object()


class _Channel:
    """One in-flight utterance's audio queue."""

    __slots__ = ("queue", "error")

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.error: BaseException | None = None


class CartesiaTTS(TTS):
    name = "cartesia"

    def __init__(
        self,
        *,
        api_key: str,
        ws_url: str,
        model: str = "sonic-latest",
        voice_id: str = DEFAULT_VOICE_ID,
        language: str = "en",
        sample_rate: int = SAMPLE_RATE,
        open_timeout: float = 6.0,
        max_concurrency: int = 2,
    ) -> None:
        self.api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.language = language
        self.sample_rate = sample_rate
        self.open_timeout = open_timeout
        self._socket = None
        self._reader: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._channels: dict[str, _Channel] = {}
        # Cartesia counts concurrent contexts against the account, not the
        # socket. The account allows 2; the orchestrator submits every segment
        # at once, so a three-sentence answer used to trip "concurrency limit
        # of 2" and silently fall back to the synthetic voice mid-answer.
        # Synthesis runs ~11x faster than playback (RTF 0.09), so bounding this
        # costs no latency: the next segment is ready long before it is heard.
        self._slots = asyncio.Semaphore(max(1, max_concurrency))

    # ---------------------------------------------------------------- plumbing

    async def _ensure_socket(self) -> None:
        if self._socket is not None and self._reader is not None and not self._reader.done():
            return
        async with self._connect_lock:
            if self._socket is not None and self._reader is not None and not self._reader.done():
                return
            await self._reset_socket()
            url = f"{self.ws_url}?api_key={self.api_key}&cartesia_version={API_VERSION}"
            try:
                self._socket = await ws_connect(
                    url,
                    additional_headers={"Cartesia-Version": API_VERSION},
                    max_size=None,
                    open_timeout=self.open_timeout,
                )
            except Exception as error:  # noqa: BLE001
                raise TTSError(f"cartesia connect failed: {error}") from error
            self._reader = asyncio.create_task(self._read_loop())

    def _request(self, text: str, context_id: str, final: bool) -> str:
        return json.dumps(
            {
                "model_id": self.model,
                "transcript": text,
                "voice": {"mode": "id", "id": self.voice_id},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
                "language": self.language,
                "context_id": context_id,
                "continue": not final,
                "add_timestamps": False,
            }
        )

    async def _reset_socket(self) -> None:
        socket, reader = self._socket, self._reader
        self._socket = None
        self._reader = None
        if reader is not None:
            reader.cancel()
            try:
                await reader
            except (asyncio.CancelledError, Exception):
                pass
        if socket is not None:
            try:
                await socket.close()
            except Exception:
                pass

    def _fail_all(self, error: BaseException) -> None:
        for channel in list(self._channels.values()):
            if channel.error is None:
                channel.error = error
            channel.queue.put_nowait(_END)

    async def _read_loop(self) -> None:
        """Demultiplex provider frames into the matching context queue."""
        socket = self._socket
        try:
            while socket is not None:
                raw = await socket.recv()
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                context_id = payload.get("context_id")
                if payload.get("type") == "error" or str(
                    payload.get("status_code", "")
                ).startswith(("4", "5")):
                    error = TTSError(f"cartesia error: {payload.get('error') or payload}")
                    channel = self._channels.get(str(context_id)) if context_id else None
                    if channel is not None:
                        channel.error = error
                        channel.queue.put_nowait(_END)
                    else:
                        self._fail_all(error)
                    continue
                if context_id is None:
                    continue
                channel = self._channels.get(str(context_id))
                if channel is None:
                    # Cancelled utterance: audio arriving after barge-in is dropped.
                    continue
                data = payload.get("data")
                if data:
                    channel.queue.put_nowait(data)
                if payload.get("done") is True or payload.get("type") == "done":
                    channel.queue.put_nowait(_END)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            self._fail_all(TTSError(f"cartesia receive failed: {error}"))
        finally:
            if self._socket is socket:
                self._socket = None

    # -------------------------------------------------------------------- api

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
            async for span in self._stream(text, context_id=context_id, final=final):
                yield span

    async def _stream(
        self,
        text: str,
        *,
        context_id: str,
        final: bool,
    ) -> AsyncIterator[AudioSpan]:
        await self._ensure_socket()
        if self._socket is None:
            raise TTSError("cartesia socket unavailable")

        channel = _Channel()
        self._channels[context_id] = channel
        resampler = StreamResampler(self.sample_rate, 48_000)
        try:
            try:
                async with self._send_lock:
                    await self._socket.send(self._request(text, context_id, final))
            except Exception as error:  # noqa: BLE001
                await self._reset_socket()
                raise TTSError(f"cartesia send failed: {error}") from error

            while True:
                try:
                    item = await asyncio.wait_for(channel.queue.get(), timeout=20.0)
                except TimeoutError as error:
                    await self._reset_socket()
                    raise TTSError("cartesia timed out waiting for audio") from error
                if item is _END:
                    if channel.error is not None:
                        raise channel.error
                    break
                samples = np.frombuffer(base64.b64decode(item), dtype="<i2").astype(np.int16)
                converted = resampler.process(samples)
                if converted.size:
                    yield AudioSpan(converted)
            tail = resampler.flush()
            if tail.size:
                yield AudioSpan(tail)
        finally:
            self._channels.pop(context_id, None)

    async def cancel(self, context_id: str) -> None:
        """Tell the provider to stop. The egress shaper is the real enforcement."""
        channel = self._channels.pop(context_id, None)
        if channel is not None:
            channel.queue.put_nowait(_END)
        if self._socket is None:
            return
        try:
            async with self._send_lock:
                await self._socket.send(json.dumps({"context_id": context_id, "cancel": True}))
        except Exception:
            pass

    async def close(self) -> None:
        await self._reset_socket()

    async def list_voices(self) -> list[dict]:
        import httpx

        base = self.ws_url.replace("wss://", "https://").split("/tts/", 1)[0]
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{base}/voices",
                params={"limit": 100},
                headers={"Authorization": f"Bearer {self.api_key}", "Cartesia-Version": API_VERSION},
            )
            response.raise_for_status()
            payload = response.json()
        return payload if isinstance(payload, list) else payload.get("data", [])


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def default_segments(text: str, max_chars: int = 120) -> list[str]:
    """Small helper used by probes and tests; the orchestrator has its own."""
    out: list[str] = []
    for sentence in SENTENCE_SPLIT.split(text.strip()):
        remaining = sentence.strip()
        while len(remaining) > max_chars:
            cut = remaining.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            out.append(remaining)
    return out
