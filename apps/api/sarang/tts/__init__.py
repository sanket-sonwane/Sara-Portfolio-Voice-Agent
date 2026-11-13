"""TTS provider selection and a fallback wrapper.

The wrapper is the part that matters in production. A provider can connect and
then fail mid-utterance; the turn must not die with a socket error, and the
visitor must not be left in silence.

The recovery depends on what the provider can prove:

* nothing emitted - the synthetic voice speaks the whole segment;
* emitted, with alignment - the synthetic voice speaks only the unspoken tail;
* emitted, without alignment - the spoken boundary is unknown, so the turn stops
  on the prefix it actually spoke and the metrics record the degradation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..config import Settings
from .base import TTS, AudioSpan, TTSError, TTSPartialError
from .cartesia import CartesiaTTS
from .sarvam import SarvamTTS
from .smallest import SmallestTTS
from .synthetic import SyntheticTTS

log = logging.getLogger(__name__)


class FallbackTTS(TTS):
    def __init__(self, primary: TTS, fallback: TTS | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or SyntheticTTS()
        self.name = primary.name
        self.degraded_turns = 0

    async def prepare(self, text: str) -> None:
        await self.primary.prepare(text)

    async def synthesize(
        self,
        text: str,
        *,
        context_id: str,
        final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AudioSpan]:
        emitted = False
        # Highest character offset the provider reported as covered. Only
        # meaningful while every span carried alignment.
        covered = 0
        aligned = True
        try:
            async for span in self.primary.synthesize(
                text, context_id=context_id, final=final, metadata=metadata
            ):
                emitted = True
                if span.text_range is None:
                    aligned = False
                else:
                    covered = max(covered, span.text_range[1])
                yield span
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            log.warning("tts %s failed: %s", self.primary.name, error)
            self.degraded_turns += 1
            if not emitted:
                # Nothing was spoken, so the fallback may say the whole segment.
                pass
            elif aligned and covered < len(text):
                # The provider told us exactly where it stopped. Resume only
                # the unspoken tail; repeating the prefix is worse than a
                # different voice for the tail.
                tail = text[covered:].lstrip()
                if not tail:
                    return
                async for span in self.fallback.synthesize(
                    tail, context_id=context_id, final=final, metadata=metadata
                ):
                    yield span
                return
            else:
                # No alignment: the spoken boundary is unknown. Do not replay
                # the segment and do not pretend it was completed; the turn
                # stops on the prefix it actually spoke.
                raise TTSPartialError(text, covered if aligned else 0) from error
        async for span in self.fallback.synthesize(
            text, context_id=context_id, final=final, metadata=metadata
        ):
            yield span

    async def close(self) -> None:
        for client in (self.primary, self.fallback):
            try:
                await client.close()
            except Exception:
                pass


def build_tts(settings: Settings) -> TTS:
    provider = settings.tts_provider
    if provider == "sarvam" and settings.sarvam_key:
        primary: TTS = SarvamTTS(
            api_key=settings.sarvam_key,
            base_url=settings.sarvam_base_url,
            model=settings.sarvam_tts_model,
            voice=settings.sarvam_voice,
        )
        return FallbackTTS(primary)
    if provider == "smallest" and settings.smallest_key:
        primary = SmallestTTS(
            api_key=settings.smallest_key,
            ws_url=settings.smallest_ws_url,
            model=settings.smallest_model,
            voice_id=settings.smallest_voice,
            language=settings.smallest_language,
            max_concurrency=settings.smallest_max_concurrency,
        )
        return FallbackTTS(primary)
    if provider == "cartesia" and settings.cartesia_key:
        primary = CartesiaTTS(
            api_key=settings.cartesia_key,
            ws_url=settings.cartesia_ws_url,
            model=settings.cartesia_model,
            voice_id=settings.cartesia_voice,
            max_concurrency=settings.tts_max_concurrency,
        )
        return FallbackTTS(primary)
    return SyntheticTTS()


__all__ = ["AudioSpan", "FallbackTTS", "TTSError", "TTSPartialError", "TTS", "build_tts"]
