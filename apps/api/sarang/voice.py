"""Utterance loop: VAD, ASR and turn handoff.

Both transports feed the same loop. WebRTC and the WebSocket audio fallback
differ only in how PCM arrives; endpointing, barge-in and the ASR lifecycle are
identical, which is why they are not implemented twice.

The sequence for one utterance is:

    frame -> VAD -> (speech start) -> ASR begin + pre-roll -> ... -> (silence) ->
    ASR finalize -> turn engine

Two properties matter more than latency here, because getting them wrong makes
the recogniser look broken:

* **The recogniser hears the whole utterance.** A ring buffer keeps the last
  ``ENDPOINT_PREROLL_MS`` of audio and replays it when the VAD fires, so the
  120-250 ms of speech that triggered the VAD is not lost. Frames inside the
  utterance are forwarded even when a single word dips below the energy
  threshold.
* **Finalization does not block audio.** The turn is finalized in its own task,
  so a slow ``transcript.final`` cannot stall the microphone.

The ASR socket is opened once per transport connection and reused for every
turn. The old design opened it on speech start, which spent a handshake at the
worst possible moment and forced a second buffer to guess what had been missed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .asr import build_asr
from .asr.normalize import normalize_transcript
from .audio.frames import frame_bytes, to_int16
from .audio.vad import Vad, VadConfig
from .kb.lookup import strip_discourse
from .metrics import TurnMarks
from .orchestrator import Orchestrator
from .schemas import EventType, SessionState
from .session import Session

log = logging.getLogger(__name__)

# 16 kHz mono PCM16 is 32 bytes per millisecond.
_BYTES_PER_MS = 32


@dataclass
class _Utterance:
    """State for one turn, so a late final cannot leak into the next one."""

    index: int
    latest: str = ""
    final: str = ""
    event: asyncio.Event = field(default_factory=asyncio.Event)
    pcm: bytearray = field(default_factory=bytearray)


class UtteranceLoop:
    def __init__(self, orchestrator: Orchestrator, session: Session) -> None:
        self.orchestrator = orchestrator
        self.session = session
        settings = session.settings
        self.vad = Vad(
            VadConfig(
                silence_ms=settings.endpoint_silence_ms,
                min_speech_ms=settings.endpoint_min_speech_ms,
                hangover_ms=settings.endpoint_hangover_ms,
            )
        )
        self.asr = None
        self.reader: asyncio.Task | None = None
        self.marks: TurnMarks | None = None
        self.closed = False
        self._warned = False
        self._capturing = False
        self._finalizing = False
        self._final_task: asyncio.Task | None = None
        self._utterance = 0
        self.current: _Utterance | None = None
        self._pending: dict[int, _Utterance] = {}
        # Pre-roll: only collected while no utterance is being captured, so it
        # holds the most recent 800 ms of silence and the first syllable of the
        # next question. It is replayed once, then cleared.
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0
        self._ring_limit = max(3_200, settings.endpoint_preroll_ms * _BYTES_PER_MS)
        self._dump_dir = Path(settings.asr_dump_dir) if settings.asr_dump_dir else None
        self._final_timeout = settings.asr_final_timeout_s
        self._early_silence_ms = settings.endpoint_early_silence_ms
        # Media clock and the moment the closing silence window opened. The
        # endpoint cost is measured in *media* time, not wall clock: a test (or
        # a burst replay) pushes frames faster than real time, and a wall-clock
        # difference would then report milliseconds for an 800 ms window.
        self._media_ms = 0.0
        self._silence_onset: float | None = None
        # When the recogniser last changed the partial for this utterance.
        # A stable, terminally-punctuated partial is the only evidence strong
        # enough to justify closing the turn before the full silence window.
        self._partial_changed_at = 0.0
        self._terminal_re = re.compile(r'[.!?…]["\')\]]?\s*$')

    # ------------------------------------------------------------------ input

    async def push(self, samples: np.ndarray) -> None:
        if self.closed:
            return
        pcm = samples if samples.dtype == np.int16 else to_int16(samples)
        if pcm.size == 0:
            return
        frame = frame_bytes(pcm)

        if not self._capturing:
            self._remember(frame)

        frame_start = self._media_ms
        self._media_ms += pcm.size / 16.0  # 16 samples per ms at 16 kHz
        update = self.vad.push(pcm, 16_000)

        # When the closing silence window opened, in media time. The onset is
        # stamped at the *start* of the first frame the VAD counts as silence,
        # so the measured span equals the window the detector actually waited.
        endpoint_ms = 0.0
        if update.speech_ended:
            if self._silence_onset is not None:
                endpoint_ms = max(0.0, self._media_ms - self._silence_onset)
            self._silence_onset = None
        elif self._capturing and update.silence_ms > 0.0:
            if self._silence_onset is None:
                self._silence_onset = frame_start
        elif self._capturing:
            self._silence_onset = None

        started = False
        if update.speech_started and not self._capturing:
            await self._begin_utterance()
            # ``_begin_utterance`` replays the ring, and this frame is already
            # in it. Pushing it again would duplicate 20 ms of audio at the
            # exact moment the recogniser starts listening.
            started = True

        if self._capturing and self.asr is not None and not started:
            utterance = self.current
            if utterance is not None:
                utterance.pcm.extend(frame)
            await self.asr.push(frame)

        if update.partial_due:
            utterance = self.current
            if utterance is not None and utterance.latest:
                self.session.publish(
                    EventType.TRANSCRIPT_PARTIAL,
                    {"text": utterance.latest, "source": "voice"},
                )

        if (
            not update.speech_ended
            and self._capturing
            and not self._finalizing
            and self._silence_onset is not None
        ):
            silence_ms = self._media_ms - self._silence_onset
            if self._can_close_early(silence_ms):
                # The recogniser already has a complete sentence and the
                # visitor has been silent long enough that this is very
                # unlikely to be a mid-thought pause. Close now; the VAD's
                # own 800 ms window would have reached the same conclusion
                # later. endpoint_ms is the observed silence, not the
                # configured ceiling.
                self._capturing = False
                self._start_finalize(max(0.0, silence_ms))

        if update.speech_ended and self._capturing:
            self._capturing = False
            self._start_finalize(endpoint_ms)

    def _remember(self, frame: bytes) -> None:
        self._ring.append(frame)
        self._ring_bytes += len(frame)
        while self._ring_bytes > self._ring_limit and self._ring:
            dropped = self._ring.popleft()
            self._ring_bytes -= len(dropped)

    def _can_close_early(self, silence_ms: float) -> bool:
        """True only when a shorter silence is safe evidence the turn is over.

        A partial can look finished during a mid-sentence pause (ADR 0003
        records a 460 ms pause inside a correctly-spoken question where the
        partial was already punctuated and stable). The guard therefore
        requires all of: enough observed silence (>= early threshold, and the
        threshold itself stays above that measured pause), a terminally-
        punctuated partial, and a partial that has not changed for a
        meaningful stretch - so the recogniser is not still revising it.
        Anything uncertain keeps the full 800 ms window.
        """
        early = self._early_silence_ms
        if early <= 0 or silence_ms < early:
            return False
        utterance = self.current
        if utterance is None:
            return False
        text = (utterance.latest or "").strip()
        if not text or self._terminal_re.search(text) is None:
            return False
        # Stable since at least 300 ms of media time ago. This rules out a
        # partial that just acquired its final punctuation on this frame.
        if self._partial_changed_at > self._media_ms - 300.0:
            return False
        return True

    async def _begin_utterance(self) -> None:
        session = self.session
        if session.state == SessionState.SPEAKING:
            await self.orchestrator.interrupt(session, "barge-in")

        # Speculation belongs to one utterance. An arm warmed for the
        # previous question must never be adoptable for this one; the gates
        # would likely refuse it anyway, but paying for it is pointless.
        speculator = session.speculation
        if speculator is not None:
            speculator.reset()

        if self._finalizing and self._final_task is not None:
            # A previous turn is still waiting for its final. Give it a short
            # window; if it is stuck, abandon it rather than swallow new speech.
            with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(self._final_task), timeout=1.5)
            if self._finalizing and self._final_task is not None:
                self._final_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._final_task
                self._final_task = None
                self._finalizing = False

        session.set_state(SessionState.LISTENING, "Listening")
        self._silence_onset = None

        if not await self._ensure_asr():
            return

        self._utterance += 1
        utterance = _Utterance(index=self._utterance)
        self.current = utterance
        # Sarvam's utterance counter is zero-based for the session.
        self._pending[utterance.index - 1] = utterance
        self._capturing = True
        self._partial_changed_at = self._media_ms

        await self.asr.begin_utterance()
        preroll = list(self._ring)
        self._ring.clear()
        self._ring_bytes = 0
        for buffered in preroll:
            utterance.pcm.extend(buffered)
            await self.asr.push(buffered)

    # -------------------------------------------------------------------- ASR

    async def warm(self) -> None:
        """Open the recogniser before the visitor speaks.

        The socket handshake is ~200-400 ms. Doing it on the first ``push``
        puts that delay inside the first question; doing it when the transport
        opens puts it in the connection phase, where nobody notices.
        """
        if self.closed:
            return
        with contextlib.suppress(Exception):
            await self._ensure_asr()

    async def _ensure_asr(self) -> bool:
        if self.asr is not None and getattr(self.asr, "alive", True):
            return True
        if self.asr is not None:
            await self._close_asr()

        asr = build_asr(self.session.settings)
        if getattr(asr, "name", "") == "none":
            if not self._warned:
                self._warned = True
                self.session.notice(
                    "Speech recognition isn't configured in this environment. "
                    "You can still type a question.",
                    "warning",
                )
            await asr.close()
            return False

        connect = getattr(asr, "connect", None)
        if connect is not None:
            try:
                await asyncio.wait_for(connect(), timeout=4.0)
            except Exception as error:  # noqa: BLE001
                log.warning("asr connect failed: %s", error)
                await asr.close()
                if not self._warned:
                    self._warned = True
                    self.session.notice(
                        "I couldn't open the speech recogniser. Type your question instead.",
                        "warning",
                    )
                return False

        self.asr = asr
        self.reader = asyncio.create_task(self._read_asr())
        return True

    async def _read_asr(self) -> None:
        assert self.asr is not None
        try:
            async for event in self.asr.events():
                if not event.text:
                    continue
                utterance = self._pending.get(event.utterance) or self.current
                if utterance is None:
                    continue
                if event.final:
                    utterance.final = normalize_transcript(event.text)
                    utterance.event.set()
                    # Keep reading: one socket serves the whole conversation.
                    continue
                text = normalize_transcript(event.text)
                if text != utterance.latest:
                    utterance.latest = text
                    self._partial_changed_at = self._media_ms
                if utterance is self.current:
                    self.session.publish(
                        EventType.TRANSCRIPT_PARTIAL,
                        {"text": utterance.latest, "source": "voice", "final": False},
                    )
                    # Warm the lookup (and the speculative arm) from the
                    # question itself, never from the discourse wrapped
                    # around it - the same strip the turn engine applies.
                    # Awaited inline: it is an in-memory lookup, and keeping
                    # it in the reader preserves partial ordering.
                    speculator = self.session.speculation
                    if speculator is not None:
                        await speculator.note(strip_discourse(utterance.latest))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            log.warning("asr stream failed: %s", error)
        finally:
            # Unblock any turn waiting on a final that will never arrive.
            for utterance in list(self._pending.values()):
                utterance.event.set()

    async def _close_asr(self) -> None:
        if self.reader is not None:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.reader
            self.reader = None
        if self.asr is not None:
            with contextlib.suppress(Exception):
                await self.asr.close()
            self.asr = None

    # --------------------------------------------------------------- finalize

    def _start_finalize(self, endpoint_ms: float = 0.0) -> None:
        if self._finalizing:
            return
        self._finalizing = True
        self._final_task = asyncio.create_task(self._finalize(endpoint_ms))

    async def _finalize(self, endpoint_ms: float = 0.0) -> None:
        session = self.session
        utterance = self.current
        marks = TurnMarks("pending")
        marks.mark_end_of_speech()
        marks.mark_endpoint(endpoint_ms)
        marks.source = "voice"
        session.set_state(SessionState.TRANSCRIBING, "Transcribing")
        try:
            if utterance is None:
                return
            marks.enter("asr_final")
            if self.asr is not None:
                with contextlib.suppress(Exception):
                    await self.asr.finalize()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        utterance.event.wait(), timeout=self._final_timeout
                    )
            asr_final_ms = marks.exit("asr_final")

            text = (utterance.final or utterance.latest).strip()
            source = "final" if utterance.final else ("partial" if utterance.latest else "none")
            if not utterance.final and text:
                log.warning(
                    "asr final timed out after %.1fs; using partial transcript (%d chars)",
                    self._final_timeout,
                    len(text),
                )
            self._pending.pop(utterance.index - 1, None)
            if self.current is utterance:
                self.current = None

            await self._dump(utterance, text, source, asr_final_ms)

            if not text:
                if session.state != SessionState.ERROR:
                    session.notice("I didn't catch that. Try again, or type it.", "info")
                    session.set_state(SessionState.LISTENING)
                return

            session.publish(
                EventType.TRANSCRIPT_FINAL,
                {"text": text, "source": "voice"},
            )
            task = asyncio.create_task(
                self.orchestrator.run_turn(session, text, marks=marks, source="voice")
            )
            session.turn_task = task
        finally:
            self._finalizing = False
            self._final_task = None

    # ------------------------------------------------------------------- dump

    async def _dump(
        self,
        utterance: _Utterance,
        text: str,
        source: str,
        asr_final_ms: float | None,
    ) -> None:
        if self._dump_dir is None or not utterance.pcm:
            return
        directory = self._dump_dir
        index = utterance.index
        payload = {
            "utterance": index,
            "text": text,
            "source": source,
            "partial": utterance.latest,
            "final": utterance.final,
            "asr": getattr(self.asr, "server_config", {}) if self.asr else {},
            "asr_error": getattr(self.asr, "last_error", "") if self.asr else "",
            "asr_final_ms": round(asr_final_ms, 1) if asr_final_ms is not None else None,
            "bytes": len(utterance.pcm),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        pcm = bytes(utterance.pcm)
        try:
            await asyncio.to_thread(self._write_dump, directory, index, pcm, payload)
        except Exception as error:  # noqa: BLE001 - diagnostics must never break a turn
            log.debug("asr dump failed: %s", error)

    @staticmethod
    def _write_dump(directory: Path, index: int, pcm: bytes, payload: dict) -> None:
        import wave

        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        base = directory / f"utt-{index:03d}-{stamp}"
        with wave.open(str(base.with_suffix(".wav")), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(pcm)
        base.with_suffix(".json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    # ---------------------------------------------------------------- lifetime

    async def close(self) -> None:
        self.closed = True
        if self._final_task is not None:
            self._final_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._final_task
            self._final_task = None
        await self._close_asr()
        self.vad.reset()
