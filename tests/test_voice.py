"""Utterance loop: pre-roll, one turn per utterance, no mid-sentence split.

These are the regressions behind "it starts talking while I am still talking".
The recogniser itself is faked; what is under test is the audio the loop decides
to send and when it decides a turn is over.
"""

from __future__ import annotations

import asyncio

import pytest
from sarang.asr.base import ASREvent, ASRSession
from sarang.bus import Bus
from sarang.config import resolve_settings
from sarang.orchestrator import Orchestrator
from sarang.session import SessionRegistry
from sarang.voice import UtteranceLoop

from tests.conftest import silence, vocal


class FakeASR(ASRSession):
    name = "fake"

    def __init__(self) -> None:
        self.pushed: list[bytes] = []
        self.begins = 0
        self.finalizes = 0
        self.closed = False
        self.text = "what did you build at arai"
        self._events: asyncio.Queue[ASREvent | None] = asyncio.Queue()

    async def begin_utterance(self) -> None:
        self.begins += 1

    async def push(self, pcm: bytes) -> None:
        self.pushed.append(pcm)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def finalize(self) -> None:
        self.finalizes += 1
        await self._events.put(
            ASREvent(text=self.text, final=True, utterance=self.begins - 1)
        )

    async def close(self) -> None:
        self.closed = True
        await self._events.put(None)


class SilentFinalASR(FakeASR):
    """Never emits a final: the loop must fall back to the partial, and say so."""

    async def finalize(self) -> None:
        self.finalizes += 1


@pytest.fixture
async def engine():
    settings = resolve_settings(
        {
            "SESSION_SECRET": "test-secret",
            "SARANG_PROFILE": "local",
            "SARANG_ENV": "test",
            "ENDPOINT_SILENCE_MS": "800",
            "ENDPOINT_MIN_SPEECH_MS": "120",
            "ENDPOINT_HANGOVER_MS": "250",
            "ENDPOINT_PREROLL_MS": "800",
            "ASR_FINAL_TIMEOUT_S": "0.2",
        }
    )
    bus = Bus()
    registry = SessionRegistry(settings, bus)
    orchestrator = Orchestrator(settings, registry)
    await orchestrator.startup()
    try:
        yield orchestrator
    finally:
        await orchestrator.shutdown()


@pytest.fixture
def fake_asr(monkeypatch):
    asr = FakeASR()
    monkeypatch.setattr("sarang.voice.build_asr", lambda settings: asr)
    return asr


@pytest.fixture
def no_turn(engine, monkeypatch):
    async def run_turn(*args, **kwargs):
        return None

    monkeypatch.setattr(engine, "run_turn", run_turn)


async def test_preroll_keeps_the_first_syllable(engine, fake_asr, no_turn):
    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        # A real transport pushes room tone before the visitor speaks; the
        # pre-roll must start from the very first frame of the connection.
        frames = [silence(20) for _ in range(6)]
        frames += [vocal(20, offset_ms=i * 20.0) for i in range(8)]
        for frame in frames:
            await loop.push(frame)

        assert fake_asr.begins == 1
        # Every frame from before the VAD fired is replayed, in order, starting
        # with the very first one. The old loop dropped the first 180 ms.
        assert fake_asr.pushed[0] == frames[0].tobytes()
        # The frame that triggered the VAD is in the pre-roll and must not be
        # sent twice.
        assert len(fake_asr.pushed) == len(frames)
    finally:
        await loop.close()


async def test_mid_sentence_pause_does_not_split_the_turn(engine, fake_asr, no_turn):
    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        # 400 ms of thinking time: below the 800 ms endpoint window, above the
        # 250 ms hangover. This used to end the turn and start an answer.
        for _ in range(20):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=(200.0 + i * 20.0)))

        await asyncio.sleep(0)
        assert fake_asr.begins == 1
        assert fake_asr.finalizes == 0

        # 250 ms hangover + 800 ms silence, with a margin.
        for _ in range(60):
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)
        assert fake_asr.finalizes == 1
    finally:
        await loop.close()


async def test_partial_is_only_used_when_the_final_never_arrives(
    engine, monkeypatch, no_turn
):
    asr = SilentFinalASR()
    monkeypatch.setattr("sarang.voice.build_asr", lambda settings: asr)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        await asr._events.put(
            ASREvent(text="what did you build", final=False, utterance=0)
        )
        await asyncio.sleep(0)
        assert loop.current is not None
        assert loop.current.latest == "what did you build"

        for _ in range(60):
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)
        # The partial was used, but only after the final timed out.
        assert asr.finalizes == 1
    finally:
        await loop.close()
