"""Pre-rendered acknowledgements.

A filler is only allowed on a slow path, and it must be cheap when it fires: if
synthesising the filler costs as much as the answer, it has made the turn worse.
So the bank is rendered once at startup, cached to disk, and served as PCM.

Rules enforced here:

* Never used for a turn that is already fast.
* Never spoken as a fact — these are all pure acknowledgements.
* Never allowed to delay the real answer: the turn engine keeps working while the
  filler plays, and the answer is queued behind it in the same generation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

import numpy as np

from .base import TTS
from .synthetic import render as render_synthetic

log = logging.getLogger(__name__)

PHRASES = (
    "One moment.",
    "Let me pull that up.",
    "Checking the case study.",
    "Give me a second.",
)


class FillerBank:
    def __init__(
        self,
        tts: TTS,
        cache_dir: Path,
        *,
        enabled: bool = True,
        voice_tag: str = "",
        phrases: tuple[str, ...] = PHRASES,
    ) -> None:
        self.tts = tts
        # A cached clip is only valid for the voice that rendered it. Without
        # this tag, changing CARTESIA_VOICE_ID leaves the old voice speaking
        # the fillers while the answer arrives in the new one.
        tag = "".join(char if char.isalnum() else "-" for char in voice_tag.lower()).strip("-")
        self.cache_dir = cache_dir / tag if tag else cache_dir
        self.enabled = enabled
        self.phrases = phrases
        self._memory: dict[str, np.ndarray] = {}
        self._locks: dict[str, asyncio.Lock] = {phrase: asyncio.Lock() for phrase in phrases}
        self._index = 0
        self.failures = 0

    # ------------------------------------------------------------------ render

    def _path(self, phrase: str) -> Path:
        safe = "".join(char if char.isalnum() else "_" for char in phrase.lower())
        return self.cache_dir / f"{safe}.npy"

    async def _render(self, phrase: str) -> np.ndarray | None:
        async with self._locks[phrase]:
            if phrase in self._memory:
                return self._memory[phrase]
            path = self._path(phrase)
            if path.exists():
                try:
                    audio = np.load(path)
                    self._memory[phrase] = audio.astype(np.int16)
                    return self._memory[phrase]
                except Exception:
                    pass
            audio = np.zeros(0, dtype=np.int16)
            try:
                parts: list[np.ndarray] = []
                async for span in self.tts.synthesize(
                    phrase, context_id=f"filler:{abs(hash(phrase)) % 10_000}", final=True
                ):
                    parts.append(span.pcm)
                if parts:
                    audio = np.concatenate(parts)
            except Exception as error:  # noqa: BLE001 - a filler may never break a turn
                self.failures += 1
                log.debug("filler render failed for %r: %s", phrase, error)
            if audio.size == 0:
                # The fallback is short and obviously non-verbal; still better
                # than leaving the visitor in silence on a slow lookup.
                audio = render_synthetic(phrase)
            if audio.size:
                self._memory[phrase] = audio
                with contextlib.suppress(Exception):
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(path, audio)
                return audio
            return None

    async def warm(self) -> None:
        if not self.enabled:
            return
        for phrase in self.phrases:
            with contextlib.suppress(Exception):
                await self._render(phrase)

    def warm_background(self) -> None:
        if not self.enabled:
            return
        asyncio.create_task(self.warm())

    # -------------------------------------------------------------------- read

    def pick(self) -> str:
        """Round-robin so a visitor does not hear the same phrase twice in a row."""
        phrase = self.phrases[self._index % len(self.phrases)]
        self._index += 1
        return phrase

    async def get(self, phrase: str | None = None, *, timeout: float = 1.5) -> np.ndarray | None:
        if not self.enabled:
            return None
        target = phrase or self.pick()
        if target in self._memory:
            return self._memory[target]
        try:
            return await asyncio.wait_for(self._render(target), timeout=timeout)
        except TimeoutError:
            return None

    def cached(self) -> dict[str, int]:
        return {phrase: int(audio.size) for phrase, audio in self._memory.items()}
