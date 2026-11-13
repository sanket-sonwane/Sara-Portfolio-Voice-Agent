"""Sarvam Bulbul streaming TTS.

Bulbul's public surface is a single POST that returns base64-encoded audio per
input string. That is not a socket, but it is still streaming from the caller's
point of view: the turn engine sends one short segment and starts playing its
audio while the next segment is being synthesised.

Audio is requested at 22.05 kHz and converted once, with a stateful resampler, to
the 48 kHz track. Asking for 24 kHz where available would shave a few
milliseconds; the voice quality at 22.05 kHz is the better trade for a portfolio.
"""

from __future__ import annotations

import base64
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

import httpx
import numpy as np

from ..audio.frames import StreamResampler
from .base import TTS, AudioSpan, TTSError

SAMPLE_RATE = 22_050


def _decode_audio(payload: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(payload)
    if raw[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        if width != 2:
            raise TTSError(f"sarvam returned {width * 8}-bit audio; expected 16-bit")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return samples, rate
    # Some API versions return headerless PCM16 at the requested rate.
    return np.frombuffer(raw, dtype="<i2").astype(np.int16), SAMPLE_RATE


class SarvamTTS(TTS):
    name = "sarvam"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = "bulbul:v4-flash",
        voice: str = "anushka",
        language: str = "en-IN",
        sample_rate: int = SAMPLE_RATE,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.language = language
        self.sample_rate = sample_rate
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                headers={"api-subscription-key": self.api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def url(self) -> str:
        return f"{self.base_url}/text-to-speech"

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "target_language_code": self.language,
            "speaker": self.voice,
            "model": self.model,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": True,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.0,
        }

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
        try:
            client = await self._get_client()
            response = await client.post(self.url, json=self._payload(text))
            if response.status_code >= 400:
                body = response.text[:300]
                raise TTSError(f"sarvam tts HTTP {response.status_code}: {body}")
            data = response.json()
        except httpx.HTTPError as error:
            raise TTSError(f"sarvam tts transport error: {error}") from error

        audios = data.get("audios") or ([data["audio"]] if data.get("audio") else [])
        if not audios:
            raise TTSError("sarvam tts returned no audio")
        for encoded in audios:
            samples, rate = _decode_audio(encoded)
            # One resampler per audio blob. A stateful filter shared across
            # concurrent segments mixes their histories and is not a stream.
            resampler = StreamResampler(rate, 48_000)
            converted = resampler.process(samples)
            tail = resampler.flush()
            if tail.size:
                converted = np.concatenate([converted, tail])
            if converted.size:
                # Yield in ~40 ms spans so the shaper can start early and a
                # cancellation never waits for a whole paragraph of audio.
                step = 1_920
                for start in range(0, len(converted), step):
                    yield AudioSpan(converted[start:start + step])
