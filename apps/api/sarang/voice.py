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
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .asr import build_asr
from .asr.base import ASREvent
from .asr.normalize import normalize_transcript
from .audio.frames import frame_bytes, to_int16
from .audio.vad import VadConfig, build_vad
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
    # The index the provider used for this turn, once one of its events has
    # been seen. ``index - 1`` is only a provisional guess: nothing in the
    # protocol guarantees the provider is zero-based or that it counts every
    # ``speech_start`` we sent.
    provider_index: int | None = None
    closed: bool = False
    # The visitor began a new utterance before this one's final arrived. The
    # transcript is still dumped for diagnostics, but it must not start an
    # answer over the top of the new question.
    superseded: bool = False


class UtteranceLoop:
    def __init__(self, orchestrator: Orchestrator, session: Session) -> None:
        self.orchestrator = orchestrator
        self.session = session
        settings = session.settings
        self.vad = build_vad(
            VadConfig(
                threshold_dbfs=settings.vad_threshold_dbfs,
                adaptive_margin_db=settings.vad_adaptive_margin_db,
                silence_ms=settings.endpoint_silence_ms,
                min_speech_ms=settings.endpoint_min_speech_ms,
                hangover_ms=settings.endpoint_hangover_ms,
                silero_threshold=settings.vad_silero_threshold,
            ),
            mode=settings.vad_mode,
            model_path=settings.vad_silero_model or None,
        )
        self.asr = None
        self.reader: asyncio.Task | None = None
        self._asr_lock = asyncio.Lock()
        self._rewarm_task: asyncio.Task | None = None
        self.marks: TurnMarks | None = None
        self.closed = False
        self._warned = False
        self._capturing = False
        self._finalizing = False
        self._final_task: asyncio.Task | None = None
        self._utterance = 0
        self.current: _Utterance | None = None
        # Explicit routing. Provider events carry an index we did not issue, so
        # the map is built as events arrive and an event that cannot be placed
        # unambiguously is dropped rather than applied to ``current``.
        self._open: list[_Utterance] = []
        self._by_provider: dict[int, _Utterance] = {}
        self._closed_indices: deque[int] = deque(maxlen=64)
        # Speculation runs off this task, never on the ASR reader: a tier-2
        # lookup inside the reader delayed ``transcript.final`` by up to 500 ms.
        self._note_text: str | None = None
        self._note_task: asyncio.Task | None = None
        # Pre-roll: only collected while no utterance is being captured, so it
        # holds the most recent 800 ms of silence and the first syllable of the
        # next question. It is replayed once, then cleared.
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0
        self._ring_limit = max(3_200, settings.endpoint_preroll_ms * _BYTES_PER_MS)
        self._dump_dir = Path(settings.asr_dump_dir) if settings.asr_dump_dir else None
        self._final_timeout = settings.asr_final_timeout_s
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
        elif self._capturing and (update.silence_ms > 0.0 or not update.speech):
            # Track silence onset on any non-speech frame while capturing, even
            # during hangover (where silence_ms may still be 0), so early
            # content-based closing can react without waiting for timer windows.
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

        if update.speech_ended and self._capturing:
            # The only place capture closes. Content-based early closing was
            # removed: it closed at 120 ms of silence while the detector was
            # still ``active``, and the VAD cannot re-fire ``speech_started``
            # for the same utterance, so every frame after the pause was
            # dropped (or replayed into the next question). Measured on
            # bench/audio/question-16k.wav, whose only internal pause is
            # 460 ms - see config.endpoint_early_silence_ms.
            self._capturing = False
            # The window is measured in media time; the turn origin is where it
            # started. Stamping it here (not inside the finalize task) is what
            # keeps the endpoint out of the ``asr_final`` number.
            decision_at = time.perf_counter()
            self._start_finalize(endpoint_ms, decision_at - endpoint_ms / 1000.0)

    def _remember(self, frame: bytes) -> None:
        self._ring.append(frame)
        self._ring_bytes += len(frame)
        while self._ring_bytes > self._ring_limit and self._ring:
            dropped = self._ring.popleft()
            self._ring_bytes -= len(dropped)

    async def _begin_utterance(self) -> None:
        session = self.session
        # Any state where a turn is actively producing something is a barge-in:
        # speaking audio, waiting on the model, or running a tool. Only the
        # synchronous half runs here (generation gate + EPOCH); provider cleanup
        # is scheduled, because this coroutine is inside the microphone path.
        if session.state in (
            SessionState.SPEAKING,
            SessionState.THINKING,
            SessionState.TOOL_WORKING,
        ):
            self.orchestrator.interrupt_soon(session, "barge-in")

        # Speculation belongs to one utterance. An arm warmed for the
        # previous question must never be adoptable for this one; the gates
        # would likely refuse it anyway, but paying for it is pointless.
        speculator = session.speculation
        if speculator is not None:
            speculator.reset()

        if self._finalizing and self._final_task is not None:
            # A previous turn is still waiting for its final. This is the
            # barge-in case: the visitor is already talking over it. Waiting
            # here - even 250 ms - stalls the microphone at the exact moment
            # speech resumes, so the old turn is abandoned synchronously and
            # its cleanup runs in its own task.
            if self.current is not None:
                # Whatever that turn produces, it must not start an answer now.
                self.current.superseded = True
            self._abandon_finalize()

        session.set_state(SessionState.USER_SPEAKING, "Hearing you")
        self._silence_onset = None

        if not await self._ensure_asr():
            # Without a reset the detector stays ``active`` and cannot emit
            # ``speech_started`` again, so the rest of this utterance (and the
            # next one, if the visitor keeps talking) would be ignored.
            self.vad.reset()
            session.set_state(SessionState.LISTENING, "Listening")
            return

        self._utterance += 1
        utterance = _Utterance(index=self._utterance)
        self.current = utterance
        self._open.append(utterance)
        # Provisional: Sarvam's counter is zero-based *in the dumps we have*,
        # but the first event rebinds this if the provider disagrees.
        self._by_provider[utterance.index - 1] = utterance
        self._capturing = True
        self._partial_changed_at = self._media_ms

        try:
            armed = await self.asr.begin_utterance()
        except Exception:  # noqa: BLE001 - a dead socket must not freeze capture
            log.warning("asr begin_utterance failed", exc_info=True)
            armed = False
        if armed is False:
            # The provider socket is not armed. Do not wait for it inside the
            # ingest path: abandon this onset and let the detector fire again.
            log.warning("asr not ready; abandoning utterance onset")
            self._capturing = False
            self._close_utterance(utterance)
            self.vad.reset()
            session.set_state(SessionState.LISTENING, "Listening")
            return

        preroll = list(self._ring)
        self._ring.clear()
        self._ring_bytes = 0
        if preroll:
            # One send for the whole ring: the recogniser buffers and batches
            # internally, and 15 awaited sends sat in the microphone path for no
            # reason.
            for buffered in preroll:
                utterance.pcm.extend(buffered)
            await self.asr.push(b"".join(preroll))

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
        async with self._asr_lock:
            return await self._ensure_asr_locked()

    async def _ensure_asr_locked(self) -> bool:
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
                await asyncio.wait_for(connect(), timeout=9.0)
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

    def _utterance_for(self, event: ASREvent) -> _Utterance | None:
        """Map a provider event onto the turn it belongs to, or drop it.

        The provider's ``utterance_idx`` is not ours: nothing guarantees it is
        zero-based, that it counts every ``speech_start``, or that it cannot
        arrive late. The old code fell back to ``self.current`` on a miss,
        which could apply the previous question's final to the next turn and
        release it early with the wrong text. Here a miss is only resolved when
        exactly one turn is open; otherwise the event is dropped and logged.
        """
        found = self._by_provider.get(event.utterance)
        if found is not None and not found.closed:
            return found
        if event.utterance in self._closed_indices:
            log.debug(
                "asr %s for closed turn idx=%s; ignoring",
                "final" if event.final else "partial",
                event.utterance,
            )
            return None
        open_turns = [item for item in self._open if not item.closed]
        if len(open_turns) == 1:
            utterance = open_turns[0]
            utterance.provider_index = event.utterance
            self._by_provider[event.utterance] = utterance
            return utterance
        log.warning(
            "asr %s for unplaceable utterance_idx=%s (open turns=%d); dropping",
            "final" if event.final else "partial",
            event.utterance,
            len(open_turns),
        )
        return None

    def _close_utterance(self, utterance: _Utterance) -> None:
        utterance.closed = True
        for key, value in list(self._by_provider.items()):
            if value is utterance:
                self._closed_indices.append(key)
                del self._by_provider[key]
        with contextlib.suppress(ValueError):
            self._open.remove(utterance)
        if self.current is utterance:
            self.current = None

    def _schedule_note(self, text: str) -> None:
        """Warm speculation off the ASR reader.

        ``Speculator.note`` can run a tier-2 retrieval with a 500 ms timeout.
        Awaiting it inside the reader delayed delivery of ``transcript.final``
        by exactly that much, which then showed up as ``asr_final``. This
        coalesces bursts: while one note is running, only the newest partial
        is kept.
        """
        if self.session.speculation is None:
            return
        self._note_text = text
        if self._note_task is None or self._note_task.done():
            self._note_task = asyncio.create_task(self._drain_notes())

    async def _drain_notes(self) -> None:
        while True:
            text = self._note_text
            self._note_text = None
            if text is None:
                return
            speculator = self.session.speculation
            if speculator is None:
                return
            try:
                await speculator.note(strip_discourse(text))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - speculation must never break a turn
                log.debug("speculation note failed", exc_info=True)

    async def _read_asr(self) -> None:
        assert self.asr is not None
        try:
            async for event in self.asr.events():
                if not event.text:
                    continue
                utterance = self._utterance_for(event)
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
                    self._schedule_note(utterance.latest)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            log.warning("asr stream failed: %s", error)
        finally:
            # Unblock any turn waiting on a final that will never arrive.
            for utterance in list(self._open):
                utterance.event.set()
            # Reconnect in the background, not on the next ``push``: rebuilding
            # the socket can take seconds, and the ingest path must never pay
            # it while the visitor is already talking.
            if not self.closed:
                self._rewarm_task = asyncio.create_task(self._rewarm_asr())

    async def _rewarm_asr(self) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.sleep(0.2)
            if not self.closed:
                await self._ensure_asr()

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

    def _abandon_finalize(self) -> None:
        """Stop waiting for a previous turn's final without blocking the mic.

        The task is cancelled, not awaited: its ``finally`` closes the
        utterance. Nothing it produces can run a turn because the utterance was
        already marked ``superseded``.
        """
        task, self._final_task = self._final_task, None
        self._finalizing = False
        if task is not None and not task.done():
            task.cancel()

    def _start_finalize(self, endpoint_ms: float = 0.0, origin: float | None = None) -> None:
        if self._finalizing:
            return
        self._finalizing = True
        # Capture the turn now: by the time the coroutine runs, ``current`` may
        # already be the next utterance (barge-in), and finalizing that one with
        # this turn's audio would be the exact misroute this loop just fixed.
        utterance = self.current
        self._final_task = asyncio.create_task(
            self._finalize(endpoint_ms, origin, utterance)
        )

    async def _finalize(
        self,
        endpoint_ms: float = 0.0,
        origin: float | None = None,
        utterance: _Utterance | None = None,
    ) -> None:
        session = self.session
        # A voice turn is measured from the last speech frame, not from the
        # moment this coroutine was scheduled: the gap between the two is the
        # endpoint window, and it used to be charged to ASR.
        marks = TurnMarks("pending", origin=origin or time.perf_counter())
        marks.mark_end_of_speech(marks.origin)
        marks.mark_endpoint(endpoint_ms)
        marks.source = "voice"
        if self.current is utterance:
            session.set_state(SessionState.TRANSCRIBING, "Transcribing")
        try:
            if utterance is None:
                return
            # Overlap: while waiting for the provider's final transcript,
            # warm/adopt speculation from the latest partial so lookup and a
            # speculative LLM arm can run during ASR finalization instead of
            # after it. If the final differs, consume()'s gates refuse it.
            # Debounced and off the reader: see ``_schedule_note``.
            if utterance.latest:
                self._schedule_note(utterance.latest)
            if self.asr is not None:
                marks.enter("asr_finalize")
                with contextlib.suppress(Exception):
                    await self.asr.finalize()
                marks.exit("asr_finalize")
                marks.enter("asr_final")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        utterance.event.wait(), timeout=self._final_timeout
                    )
                asr_final_ms = marks.exit("asr_final")
            else:
                asr_final_ms = None

            text = (utterance.final or utterance.latest).strip()
            source = "final" if utterance.final else ("partial" if utterance.latest else "none")
            if not utterance.final and text:
                log.warning(
                    "asr final timed out after %.1fs; using partial transcript (%d chars)",
                    self._final_timeout,
                    len(text),
                )

            await self._dump(utterance, text, source, asr_final_ms)

            if not text:
                if self.current is utterance and session.state != SessionState.ERROR:
                    session.notice("I didn't catch that. Try again, or type it.", "info")
                    session.set_state(SessionState.LISTENING)
                return

            if utterance.superseded:
                # The visitor is already asking the next question. Dumping the
                # transcript keeps it recoverable; publishing it or running a
                # turn would put the previous answer on top of them.
                log.info(
                    "dropping superseded transcript for utterance %d (%d chars)",
                    utterance.index,
                    len(text),
                )
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
            # Unconditional: a cancelled finalize must not leave the utterance
            # in ``_open`` / ``_by_provider``, where a late provider event could
            # still be routed to a turn that will never run.
            self._close_utterance(utterance)
            if self._final_task is asyncio.current_task():
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
            "provider_index": utterance.provider_index,
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
        if self._note_task is not None:
            self._note_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._note_task
            self._note_task = None
        if self._rewarm_task is not None:
            self._rewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._rewarm_task
            self._rewarm_task = None
        await self._close_asr()
        self.vad.reset()
