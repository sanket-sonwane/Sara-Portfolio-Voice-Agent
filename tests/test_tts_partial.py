"""Mid-segment TTS failure: resume honestly or stop honestly.

A provider that fails after emitting audio leaves one question open: where did
it stop? The answer decides what the visitor hears next.

* With alignment, the wrapper resumes only the unspoken tail.
* Without alignment, the boundary is unknown, so the segment is not committed
  and the turn ends on what was actually spoken.
* With nothing emitted, the whole segment may be said by the fallback voice.

These tests pin all three, because the previous behaviour - silently returning
after a partial segment - left a sentence cut in half with no record of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from sarang.tts import FallbackTTS, TTSPartialError
from sarang.tts.base import TTS, AudioSpan

TEXT = "hello there"


class _Primary(TTS):
    name = "primary"

    def __init__(self, spans: list[AudioSpan], error: Exception | None = None) -> None:
        self.spans = spans
        self.error = error
        self.seen: list[str] = []

    async def synthesize(self, text, *, context_id, final=True, metadata=None):
        self.seen.append(text)
        for span in self.spans:
            yield span
        if self.error is not None:
            raise self.error


class _Fallback(TTS):
    name = "fallback"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def synthesize(self, text, *, context_id, final=True, metadata=None):
        self.seen.append(text)
        yield AudioSpan(pcm=np.ones(8, dtype=np.int16))


async def _drain(tts: TTS) -> list[AudioSpan]:
    spans: list[AudioSpan] = []
    async for span in tts.synthesize(TEXT, context_id="ctx"):
        spans.append(span)
    return spans


def _span(chars: int = 4, text_range: tuple[int, int] | None = None) -> AudioSpan:
    return AudioSpan(pcm=np.ones(chars, dtype=np.int16), text_range=text_range)


async def test_unaligned_partial_raises_instead_of_replaying_the_segment():
    """No alignment means the spoken boundary is unknown, so nothing is guessed."""
    primary = _Primary([_span()], RuntimeError("socket closed"))
    fallback = _Fallback()
    tts = FallbackTTS(primary, fallback)

    with pytest.raises(TTSPartialError) as caught:
        await _drain(tts)

    assert caught.value.spoken_chars == 0
    assert fallback.seen == [], "replaying the segment would repeat heard speech"
    assert tts.degraded_turns == 1


async def test_aligned_partial_resumes_only_the_unspoken_tail():
    primary = _Primary(
        [_span(text_range=(0, 5))],
        RuntimeError("provider dropped after the first word"),
    )
    fallback = _Fallback()
    tts = FallbackTTS(primary, fallback)

    spans = await _drain(tts)

    assert len(spans) == 2, "the emitted prefix is kept, then the tail is spoken"
    assert fallback.seen == ["there"]
    assert tts.degraded_turns == 1


async def test_aligned_partial_covering_the_whole_text_still_stops_honestly():
    """Full alignment is not a completion signal.

    A provider can report that a span covers the last character and still fail
    before it confirms the segment. The wrapper must not turn that into a
    committed sentence: the attempt is reported as partial at that offset.
    """
    primary = _Primary(
        [_span(text_range=(0, len(TEXT)))],
        RuntimeError("failed after the last sample"),
    )
    fallback = _Fallback()
    tts = FallbackTTS(primary, fallback)

    with pytest.raises(TTSPartialError) as caught:
        await _drain(tts)

    assert caught.value.spoken_chars == len(TEXT)
    assert fallback.seen == []


async def test_zero_emitted_uses_the_whole_fallback():
    """Nothing was heard, so the fallback may say the whole segment."""
    primary = _Primary([], RuntimeError("connect failed"))
    fallback = _Fallback()
    tts = FallbackTTS(primary, fallback)

    spans = await _drain(tts)

    assert fallback.seen == [TEXT]
    assert len(spans) == 1
    assert tts.degraded_turns == 1


async def test_a_failed_fallback_is_not_silently_swallowed():
    """A fallback failure before any audio is an ordinary provider error."""

    class _BrokenFallback(_Fallback):
        async def synthesize(self, text, *, context_id, final=True, metadata=None):
            self.seen.append(text)
            raise RuntimeError("also down")
            if False:  # pragma: no cover - keeps this an async generator
                yield None

    tts = FallbackTTS(_Primary([], RuntimeError("primary down")), _BrokenFallback())

    with pytest.raises(RuntimeError):
        await _drain(tts)
    assert tts.degraded_turns == 1
