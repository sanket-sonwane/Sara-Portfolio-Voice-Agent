"""A dependency-free fallback voice.

This is not a speech synthesiser and it does not pretend to be. It produces
voice-shaped audio with the right duration, pitch movement and pauses, so the
entire real-time path — framing, pacing, barge-in, diagnostics — can be measured
with zero API keys and with no model download.

The transcript in the UI is the source of truth. This exists so that latency and
interruption are testable before a provider account exists, and so a provider
outage degrades to *something audible* instead of silence.
"""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from .base import TTS, AudioSpan

SAMPLE_RATE = 48_000
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Rough first/second formant pairs for five vowels. Values are Hz.
_VOWELS = (
    (730.0, 1090.0),
    (530.0, 1840.0),
    (270.0, 2290.0),
    (570.0, 840.0),
    (300.0, 870.0),
)


def _word_audio(word: str, rng: np.random.Generator, rate: int) -> np.ndarray:
    letters = sum(1 for char in word if char.isalpha()) or len(word) or 1
    duration = min(0.42, 0.10 + 0.032 * letters)
    count = max(1, int(duration * rate))
    t = np.arange(count, dtype=np.float32) / rate

    vowel = _VOWELS[sum(ord(char) for char in word) % len(_VOWELS)]
    base_f0 = 108.0 + (sum(ord(char) for char in word) % 37)
    # A gentle falling contour across the word.
    f0 = base_f0 * (1.0 - 0.10 * (t / max(duration, 1e-3)))

    signal = np.zeros(count, dtype=np.float32)
    for harmonic in range(1, 18):
        if base_f0 * harmonic > rate / 2:
            break
        frequency = f0 * harmonic
        # Resonance envelope: two broad Gaussian bumps at the formants.
        gain = np.exp(-((frequency - vowel[0]) ** 2) / (2 * 220.0 ** 2))
        gain += 0.7 * np.exp(-((frequency - vowel[1]) ** 2) / (2 * 320.0 ** 2))
        gain += 0.18 / harmonic
        signal += gain * np.sin(2 * math.pi * frequency * t + harmonic).astype(np.float32)

    # Attack/decay so words do not click into each other.
    envelope = np.minimum(1.0, np.minimum(t / 0.012, (duration - t) / 0.02))
    envelope = np.clip(envelope, 0.0, 1.0)
    signal *= envelope

    # A short consonant burst makes word boundaries audible.
    burst = min(int(0.018 * rate), count)
    if burst > 0:
        noise = rng.standard_normal(burst).astype(np.float32) * 0.10
        signal[:burst] += noise * np.linspace(1.0, 0.0, burst, dtype=np.float32) ** 2

    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.24 * 32767.0).astype(np.int16)


def render(text: str, *, rate: int = SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    """Turn text into roughly the right duration of voice-shaped tone."""
    rng = np.random.default_rng(seed)
    words = _WORD_RE.findall(text)
    if not words:
        return np.zeros(0, dtype=np.int16)
    pieces: list[np.ndarray] = []
    for index, word in enumerate(words):
        pieces.append(_word_audio(word, rng, rate))
        pause = 0.075 if word.endswith((".", "!", "?")) else 0.038
        if index == len(words) - 1:
            pause = 0.08
        pieces.append(np.zeros(int(pause * rate), dtype=np.int16))
    return np.concatenate(pieces)


class SyntheticTTS(TTS):
    name = "synthetic"

    def __init__(self, rate: int = SAMPLE_RATE) -> None:
        self.rate = rate

    async def synthesize(
        self,
        text: str,
        *,
        context_id: str,
        final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AudioSpan]:
        audio = render(text, rate=self.rate)
        step = int(self.rate * 0.04)  # 40 ms
        for start in range(0, len(audio), step):
            yield AudioSpan(audio[start:start + step])
