"""NVIDIA Nemotron ASR adapter.

The public NIM surface is an OpenAI-compatible ``/v1/audio/transcriptions``
endpoint, which is batch rather than incremental. For the ``iad-global`` profile
that is still the right adapter: the gateway co-locates with the endpoint, so the
round trip is small, and the recogniser is strong.

A true incremental path exists (Riva gRPC, or a self-hosted NIM with a streaming
API). ``stream_url`` is accepted so that a NIM we host next to the gateway can be
used without changing this class shape. Until then, hypotheses arrive at
finalization, which is exactly what the turn engine needs to start the LLM.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator

import httpx
import numpy as np

from .base import ASRError, ASREvent, ASRSession


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


class NemotronASR(ASRSession):
    name = "nemotron"

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        model: str,
        timeout: float = 20.0,
        sample_rate: int = 16_000,
        stream_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout = timeout
        self.sample_rate = sample_rate
        self.stream_url = stream_url
        self._buffer: list[np.ndarray] = []
        self._final: ASREvent | None = None
        self._closed = False

    async def push(self, pcm: bytes) -> None:
        if self._closed:
            return
        self._buffer.append(np.frombuffer(pcm, dtype="<i2").astype(np.int16))

    async def events(self) -> AsyncIterator[ASREvent]:
        if self._final is not None and self._final.text:
            yield self._final

    async def finalize(self) -> None:
        if self._closed or not self._buffer:
            return
        samples = np.concatenate(self._buffer)
        self._buffer.clear()
        files = {"file": ("utterance.wav", _wav_bytes(samples), "audio/wav")}
        data = {"model": self.model, "language": "en", "response_format": "json"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
                response = await client.post(self.url, headers=headers, data=data, files=files)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as error:
            raise ASRError(f"nemotron asr failed: {error}") from error
        text = str(payload.get("text", "")).strip()
        self._final = ASREvent(text=text, final=True, confidence=0.9)

    async def close(self) -> None:
        self._closed = True
        self._buffer.clear()
        await asyncio.sleep(0)


class NemotronStreamASR(ASRSession):
    """Reserved for a self-hosted NIM that exposes a WebSocket surface.

    Kept deliberately small: the protocol is not standardised, so this accepts
    line-delimited JSON with ``text``/``transcript`` fields and ignores anything
    else. It is only constructed when ``NVIDIA_ASR_WS_URL`` is configured.
    """

    name = "nemotron-stream"

    def __init__(self, *, url: str, api_key: str = "", open_timeout: float = 6.0) -> None:
        self.url = url
        self.api_key = api_key
        self.open_timeout = open_timeout
        self._socket = None
        self._queue: asyncio.Queue[ASREvent | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._closed = False

    async def connect(self) -> None:
        try:
            from websockets.asyncio.client import connect as ws_connect
        except ImportError:  # pragma: no cover
            from websockets import connect as ws_connect  # type: ignore
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._socket = await ws_connect(
            self.url, additional_headers=headers, max_size=None, open_timeout=self.open_timeout
        )
        self._reader = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        import json

        try:
            async for raw in self._socket:
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
                text = body.get("text") or body.get("transcript") or ""
                if not text:
                    continue
                final = bool(body.get("is_final") or body.get("final") or body.get("done"))
                await self._queue.put(ASREvent(text=str(text).strip(), final=final, confidence=0.9))
        except Exception:
            pass
        finally:
            await self._queue.put(None)

    async def push(self, pcm: bytes) -> None:
        if self._socket is not None and not self._closed:
            try:
                await self._socket.send(pcm)
            except Exception:
                pass

    async def events(self) -> AsyncIterator[ASREvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def finalize(self) -> None:
        return None

    async def close(self) -> None:
        self._closed = True
        if self._reader:
            self._reader.cancel()
        if self._socket is not None:
            try:
                await self._socket.close()
            except Exception:
                pass
