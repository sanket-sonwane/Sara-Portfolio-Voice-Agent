"""ASR adapter contract.

Streaming ASR is the one stage where the protocol genuinely differs per provider:
Sarvam speaks WebSocket, NVIDIA exposes HTTP batch plus gRPC/Riva streaming. The
interface below hides that difference behind two operations: push audio, receive
hypotheses. ``finalize`` is what the turn engine calls at end-of-speech.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class ASREvent:
    text: str
    final: bool = False
    confidence: float = 0.0
    language: str = "en"
    # Server-side utterance counter (Sarvam sends ``utterance_idx``). Zero for
    # adapters that carry one utterance per connection.
    utterance: int = 0

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "final": self.final,
            "confidence": round(self.confidence, 3),
        }


class ASRSession(ABC):
    name: str = "asr"

    async def begin_utterance(self) -> bool:
        """Mark the start of a turn on a long-lived recogniser session.

        Returns whether the session is actually armed. The default is a no-op
        for one-utterance adapters, which are ready by definition. A streaming
        adapter must return ``False`` rather than wait: this is called from the
        microphone ingest path.
        """
        return True

    @abstractmethod
    async def push(self, pcm: bytes) -> None:
        """Feed mono 16 kHz PCM16 audio."""

    @abstractmethod
    def events(self) -> AsyncIterator[ASREvent]:
        """Stream hypotheses. Must terminate after ``finalize``."""

    @abstractmethod
    async def finalize(self) -> None:
        """Signal end of utterance and flush the recogniser."""

    @abstractmethod
    async def close(self) -> None:
        """Release the connection. Safe to call twice."""


class ASRError(RuntimeError):
    pass


class NullASR(ASRSession):
    """No recogniser configured.

    The voice path stays open (the browser can still talk and see a level meter),
    and the text input path remains fully functional. This is the graceful
    degradation the PRD asks for when ASR is unavailable.
    """

    name = "none"

    def __init__(self) -> None:
        self._closed = False

    async def push(self, pcm: bytes) -> None:
        return None

    async def events(self) -> AsyncIterator[ASREvent]:
        if False:  # pragma: no cover - keeps this an async generator
            yield ASREvent(text="")
        return

    async def finalize(self) -> None:
        return None

    async def close(self) -> None:
        self._closed = True


class FixtureASR(ASRSession):
    """Cycles through a scripted list. For demos and pipeline tests only."""

    name = "fixture"

    def __init__(self, phrases: list[str] | None = None, delay: float = 0.35) -> None:
        import asyncio

        self._asyncio = asyncio
        self.phrases = phrases or [
            "What is the Agentic Commerce Gateway?",
            "Tell me about the ARAI internship.",
            "Which skills does he have around LLM fine-tuning?",
        ]
        self.index = 0
        self.delay = delay
        self._sent = False
        self._final: ASREvent | None = None

    async def push(self, pcm: bytes) -> None:
        return None

    async def events(self) -> AsyncIterator[ASREvent]:
        if self._final is None:
            return
        yield self._final

    async def finalize(self) -> None:
        phrase = self.phrases[self.index % len(self.phrases)]
        self.index += 1
        await self._asyncio.sleep(self.delay)
        self._final = ASREvent(text=phrase, final=True, confidence=0.95)

    async def close(self) -> None:
        self._final = None
