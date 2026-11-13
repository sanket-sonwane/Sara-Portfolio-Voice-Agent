"""Utterance loop: pre-roll, one turn per utterance, no mid-sentence split.

These are the regressions behind "it starts talking while I am still talking".
The recogniser itself is faked; what is under test is the audio the loop decides
to send and when it decides a turn is over.
"""

from __future__ import annotations

import asyncio
import json

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


class DroppingASR(FakeASR):
    """Drops audio the way Sarvam does when its outbox overflows.

    ``dropped_ms`` is cumulative for the connection; the loop reports only the
    delta from the start of the utterance, so this turn is not charged for the
    previous one's loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dropped_ms = 100.0
        self.drop_now = 240.0

    async def finalize(self) -> None:
        self.dropped_ms += self.drop_now
        await super().finalize()


@pytest.fixture
async def engine():
    settings = resolve_settings(
        {
            "SESSION_SECRET": "test-secret",
            "SARANG_PROFILE": "local",
            "SARANG_ENV": "test",
            # The loop tests use synthetic tones; the energy detector is the one
            # that responds to them. Silero is tested separately with an
            # injected predictor, so no test needs the ONNX file.
            "VAD_MODE": "energy",
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
        # with the very first one. The old loop dropped the first 180 ms. The
        # replay is one send now, so assert on the byte stream, not the calls.
        assert fake_asr.pushed[0].startswith(frames[0].tobytes())
        assert b"".join(fake_asr.pushed) == b"".join(frame.tobytes() for frame in frames)
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


async def test_endpoint_window_is_recorded_in_media_time(engine, fake_asr, monkeypatch):
    """The observed silence window lands on the turn as the ``endpoint`` stage.

    Frames here are pushed in a burst, far faster than real time: the stage
    must still report the media-time window the detector actually waited
    (800 ms), not the wall-clock microseconds the burst took.
    """
    captured: dict = {}

    async def run_turn(session, text, **kwargs):
        captured["marks"] = kwargs.get("marks")

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        for _ in range(60):
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        marks = captured.get("marks")
        assert marks is not None
        endpoint = marks.durations().get("endpoint")
        assert endpoint is not None
        assert round(endpoint) == 800
        # The stage is visible to the metrics store the diagnostics panel reads.
        assert "endpoint" in marks.to_payload()["durations_ms"]
    finally:
        await loop.close()


async def test_partial_transcript_warms_speculation_without_discourse(engine, fake_asr, no_turn):
    """Partials feed the speculator with the question, not the filler around it."""
    session = engine.create_session("test")
    speculator = session.speculation
    assert speculator is not None
    # This test is about *what* text reaches the speculator and that the
    # reader owns the work; the production stability gate would need a second
    # partial after a dwell, which the coalescing under test collapses. Open
    # the gate here so one note is enough.
    speculator.stable_hits = 1
    speculator.min_age_ms = 0
    noted: list[str] = []
    original = speculator.note

    async def capture(text: str):
        noted.append(text)
        return await original(text)

    speculator.note = capture  # type: ignore[method-assign]
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        for _ in range(2):
            await fake_asr._events.put(
                ASREvent(
                    text="however i mean what projects did he build",
                    final=False,
                    utterance=0,
                )
            )
        # Notes are coalesced off the reader: bursts collapse to the newest
        # partial, and the reader must not be the one waiting for the lookup.
        for _ in range(20):
            await asyncio.sleep(0)
            if session.speculation.prepared >= 1:
                break
        if loop._note_task is not None:
            await asyncio.wait_for(loop._note_task, timeout=2.0)

        assert noted[-1] == "what projects did he build"
        assert len(noted) >= 1
        assert session.speculation.prepared >= 1
    finally:
        await loop.close()


async def test_short_pause_does_not_close_the_turn(engine, fake_asr, monkeypatch):
    """A punctuated, stable partial is not evidence enough to close the turn.

    The real question clip has exactly one internal pause, 460 ms
    (bench/probe/pause_profile.py), and closing there truncates the transcript
    to "What did Sanket". Content is never allowed to shorten the window.
    """
    captured: dict = {}

    async def run_turn(session, text, **kwargs):
        captured["marks"] = kwargs.get("marks")

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        await fake_asr._events.put(
            ASREvent(text="What did he build at ARAI?", final=False, utterance=0)
        )
        for _ in range(10):
            await asyncio.sleep(0)

        # 700 ms of silence: stable, punctuated, and still inside the window.
        for _ in range(35):
            await loop.push(silence(20))
        assert loop._final_task is None
        assert loop._capturing is True

        for _ in range(5):  # reach the 800 ms window
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        marks = captured.get("marks")
        assert marks is not None
        endpoint = marks.durations().get("endpoint")
        assert endpoint is not None
        assert round(endpoint) == 800
    finally:
        await loop.close()


async def test_speech_resumed_inside_the_window_is_still_sent_to_asr(
    engine, fake_asr, monkeypatch
):
    """The regression behind "the rest of my question was ignored".

    Capture must follow the detector, not the endpoint decision. Before the
    fix, a short pause closed capture while the VAD was still ``active``; the
    VAD cannot re-fire ``speech_started`` for the same utterance, so every
    resumed frame was dropped (or replayed into the next question).
    """

    async def run_turn(session, text, **kwargs):
        return None

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        sent_before_pause = len(fake_asr.pushed)

        # 400 ms, well below the 800 ms window: the visitor is mid-thought.
        for _ in range(20):
            await loop.push(silence(20))
        assert loop._final_task is None
        assert loop._capturing is True

        # They carry on. These frames must reach the recogniser.
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        assert loop._capturing is True
        assert len(fake_asr.pushed) > sent_before_pause

        for _ in range(40):  # now they have finished
            await loop.push(silence(20))
        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        await loop.close()


async def test_final_for_a_closed_turn_is_not_applied_to_the_next_turn(
    engine, fake_asr, no_turn
):
    """A late final must be dropped, never routed to ``current``.

    The old reader did ``self._pending.get(idx) or self.current``, so a final
    arriving after its turn closed could set the next turn's transcript and
    release it early with the wrong text.
    """
    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        for _ in range(40):
            await loop.push(silence(20))
        first = loop._final_task
        assert first is not None
        await fake_asr._events.put(
            ASREvent(text="what did you build at arai", final=True, utterance=0)
        )
        await asyncio.wait_for(first, timeout=2.0)

        # Next turn opens.
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        assert loop.current is not None
        second_index = loop.current.index

        # A final for the previous turn arrives late. It must not touch the
        # new turn, which is still waiting for its own final.
        await fake_asr._events.put(
            ASREvent(text="a stale previous answer", final=True, utterance=0)
        )
        for _ in range(20):
            await asyncio.sleep(0)

        assert loop.current is not None
        assert loop.current.index == second_index
        assert loop.current.final == ""
    finally:
        await loop.close()


async def test_unpunctuated_partial_waits_for_the_full_silence_window(
    engine, fake_asr, monkeypatch
):
    """Without terminal punctuation there is no early close - only silence counts.

    A partial that trails off without sentence-ending punctuation is exactly
    the mid-thought case the 800 ms window exists to protect.
    """
    captured: dict = {}

    async def run_turn(session, text, **kwargs):
        captured["marks"] = kwargs.get("marks")

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        await fake_asr._events.put(
            ASREvent(text="What did he build at ARAI", final=False, utterance=0)
        )
        for _ in range(10):
            await asyncio.sleep(0)
        # 600 ms of silence would be enough for an early close if it were
        # allowed; without punctuation it must not fire.
        for _ in range(30):
            await loop.push(silence(20))
        assert loop._final_task is None

        for _ in range(30):  # reach the full 800 ms window (+ hangover floor)
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        marks = captured.get("marks")
        assert marks is not None
        endpoint = marks.durations().get("endpoint")
        assert endpoint is not None
        assert round(endpoint) == 800
    finally:
        await loop.close()


async def test_filler_only_punctuated_partial_does_not_close_early(
    engine, fake_asr, monkeypatch
):
    """A discourse-only fragment must not end the turn at the early floor.

    ASR can punctuate fillers ("However, I mean.") while the visitor is only
    pausing before the real question. Closing early would pass the preamble to
    the model and drop the question - the "only fillers recognised" failure.
    """
    captured: dict = {}

    async def run_turn(session, text, **kwargs):
        captured["marks"] = kwargs.get("marks")

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        # Stable, punctuated, but pure discourse - not a completed question.
        await fake_asr._events.put(
            ASREvent(text="However, I mean.", final=False, utterance=0)
        )
        for _ in range(10):
            await asyncio.sleep(0)
        # Well past the early floor; must NOT trigger an early close.
        for _ in range(30):
            await loop.push(silence(20))
        assert loop._final_task is None

        for _ in range(30):  # reach the full 800 ms window
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        marks = captured.get("marks")
        assert marks is not None
        endpoint = marks.durations().get("endpoint")
        assert endpoint is not None
        assert round(endpoint) == 800
    finally:
        await loop.close()


async def test_punctuated_substantive_statement_waits_for_the_full_window(
    engine, fake_asr, monkeypatch
):
    """A period-ending statement is not a licence to close early either.

    Punctuation is a property of the recogniser's guess, not of the visitor's
    intent; the measured failure was exactly a punctuated partial at a 460 ms
    pause. Only silence ends a turn.
    """
    captured: dict = {}

    async def run_turn(session, text, **kwargs):
        captured["marks"] = kwargs.get("marks")

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()

        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        await fake_asr._events.put(
            ASREvent(text="Tell me about ARAI.", final=False, utterance=0)
        )
        for _ in range(10):
            await asyncio.sleep(0)
        for _ in range(30):  # 600 ms: still inside the window
            await loop.push(silence(20))
        assert loop._final_task is None

        for _ in range(10):  # reach 800 ms
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        marks = captured.get("marks")
        assert marks is not None
        endpoint = marks.durations().get("endpoint")
        assert endpoint is not None
        assert round(endpoint) == 800
    finally:
        await loop.close()

async def test_slow_speculation_does_not_delay_the_final(engine, fake_asr, no_turn):
    """The reader must deliver the final while speculation is still running.

    ``Speculator.note`` can spend up to 500 ms in tier-2 retrieval. Awaiting it
    on the ASR reader put that latency in front of ``transcript.final`` and it
    was then charged as ``asr_final``.
    """

    class SlowSpeculator:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self) -> None:
            return None

        async def note(self, text: str) -> None:
            self.calls += 1
            await asyncio.sleep(1.0)

    session = engine.create_session("test")
    session.speculation = SlowSpeculator()  # type: ignore[assignment]
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()
        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))

        await fake_asr._events.put(
            ASREvent(text="what did you build", final=False, utterance=0)
        )
        for _ in range(5):
            await asyncio.sleep(0)
        assert loop.current is not None

        loop_time = asyncio.get_running_loop().time()
        await fake_asr._events.put(
            ASREvent(text="what did you build at arai", final=True, utterance=0)
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if loop.current.final:
                break

        assert loop.current.final == "what did you build at arai"
        assert asyncio.get_running_loop().time() - loop_time < 0.3
    finally:
        await loop.close()

async def test_a_turn_finalized_after_new_speech_does_not_start_an_answer(engine, monkeypatch):
    """A late final for a turn the visitor has already left must not speak.

    The loop used to wait 250 ms for that final and then run the turn anyway,
    so an answer to the previous question started on top of the new one. The
    transcript is still dumped, but no turn is started.
    """
    asr = SilentFinalASR()
    monkeypatch.setattr("sarang.voice.build_asr", lambda settings: asr)
    spoken: list[str] = []

    async def run_turn(session, text, **kwargs):
        spoken.append(text)

    monkeypatch.setattr(engine, "run_turn", run_turn)

    session = engine.create_session("test")
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()
        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        await asr._events.put(
            ASREvent(text="what did you build at arai", final=False, utterance=0)
        )
        for _ in range(10):
            await asyncio.sleep(0)
        for _ in range(45):  # reach the 800 ms window
            await loop.push(silence(20))
        assert loop._final_task is not None

        # The visitor starts the next question before this final lands.
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        assert loop.current is not None
        assert loop.current.index == 2

        # Let the old finalize time out. It must not publish a final or run.
        for _ in range(40):
            await asyncio.sleep(0)
        assert spoken == []
    finally:
        await loop.close()


async def test_barge_in_cleans_up_the_abandoned_utterance(engine, monkeypatch):
    """The abandoned turn must leave the routing tables with it.

    A cancelled finalize that only reset flags would leave the utterance in
    ``_open`` / ``_by_provider``, where a late provider event could still be
    routed to a turn that will never run.
    """
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
        for _ in range(45):  # reach the 800 ms window
            await loop.push(silence(20))

        abandoned = loop.current
        assert abandoned is not None
        assert loop._final_task is not None
        assert abandoned in loop._open
        await asyncio.sleep(0.01)  # let the finalize task reach its wait

        # New speech: the previous final is abandoned synchronously.
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        for _ in range(3):
            await asyncio.sleep(0)

        assert loop._finalizing is False
        assert abandoned not in loop._open
        assert abandoned not in loop._by_provider.values()
        assert loop.current is not None and loop.current is not abandoned
    finally:
        await loop.close()


async def test_dropped_asr_audio_is_reported_honestly(
    engine, monkeypatch, no_turn, tmp_path
):
    """Audio the recogniser shed is part of the transcript's provenance.

    Sarvam drops the oldest queued audio when its outbox overflows. The words
    in that audio were never heard by the recogniser, so a transcript that
    looks complete is a lie. The delta - not the connection's cumulative
    counter - belongs to this utterance.
    """
    asr = DroppingASR()
    monkeypatch.setattr("sarang.voice.build_asr", lambda settings: asr)

    session = engine.create_session("test")
    key, queue = engine.registry.bus.subscribe()
    loop = UtteranceLoop(engine, session)
    loop._dump_dir = tmp_path
    try:
        await loop.warm()
        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        # The baseline was 100 ms from an earlier utterance; this one loses
        # another 240 ms during finalisation.
        for _ in range(60):
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        finals = [event for event in events if event.type.value == "transcript.final"]
        assert finals
        assert finals[-1].payload["degraded"] == "asr_audio_dropped"
        assert finals[-1].payload["dropped_ms"] == 240.0
        assert finals[-1].payload["raw_text"] == "what did you build at arai"

        dumps = list(tmp_path.glob("utt-*.json"))
        assert len(dumps) == 1
        payload = json.loads(dumps[0].read_text(encoding="utf-8"))
        assert payload["degraded"] == "asr_audio_dropped"
        assert payload["dropped_ms"] == 240.0
        assert payload["raw_text"] == "what did you build at arai"
        assert payload["text"] == "what did you build at arai"
    finally:
        await loop.close()


async def test_a_clean_utterance_reports_no_degradation(engine, fake_asr, no_turn):
    """Absence of corruption must be explicit as an empty field, not a guess."""
    session = engine.create_session("test")
    key, queue = engine.registry.bus.subscribe()
    loop = UtteranceLoop(engine, session)
    try:
        await loop.warm()
        for _ in range(6):
            await loop.push(silence(20))
        for i in range(10):
            await loop.push(vocal(20, offset_ms=i * 20.0))
        for _ in range(60):
            await loop.push(silence(20))

        task = loop._final_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2.0)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        final = [event for event in events if event.type.value == "transcript.final"][-1]
        assert final.payload["degraded"] == ""
        assert final.payload["dropped_ms"] == 0.0
    finally:
        await loop.close()

