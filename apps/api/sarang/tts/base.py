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


@dataclass
class AudioSpan:
    """A block of mono 48 kHz PCM16 audio."""

    pcm: np.ndarray

    def __len__(self) -> int:
        return int(self.pcm.size)


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


class TTSError(RuntimeError):
    pass
