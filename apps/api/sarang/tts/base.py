"""TTS adapter contract.

Synthesis is per segment, not per answer: the first segment must reach the wire
while the model is still writing the rest. Adapters therefore take one short
string and stream audio, and the orchestrator decides where the boundaries are.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np


class TTSError(RuntimeError):
    pass


@dataclass
class AudioSpan:
    """A block of mono 48 kHz PCM16 audio.

    ``text_range`` is the half-open character range of the input text this
    audio covers, when the provider reports alignment. It exists for one
    reason: after a mid-segment failure, the only honest way to resume is to
    ask the provider where it stopped. Without it, a partial segment cannot be
    completed without repeating speech the visitor already heard.
    """

    pcm: np.ndarray
    text_range: tuple[int, int] | None = None

    def __len__(self) -> int:
        return int(self.pcm.size)


class TTSPartialError(TTSError):
    """Synthesis failed after some audio for the segment was already emitted.

    ``spoken_chars`` is the aligned character offset that was covered, or 0
    when the provider gave no alignment and the boundary is unknown. A caller
    must not guess in the latter case: the segment is not committed, and the
    turn stops on the prefix it actually spoke.
    """

    def __init__(self, text: str, spoken_chars: int = 0, message: str = "") -> None:
        super().__init__(message or "tts failed after partial audio")
        self.text = text
        self.spoken_chars = spoken_chars


class TTS(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        context_id: str,
        final: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[AudioSpan]:
        """Yield audio for one segment. Closing the iterator cancels synthesis."""
        raise NotImplementedError

    async def prepare(self, text: str) -> None:
        """Optional warm-up, e.g. opening a socket before the first token."""
        return None

    async def close(self) -> None:
        return None

