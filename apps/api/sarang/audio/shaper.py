"""Egress shaper.

The only writer to the outbound audio track. It guarantees four properties:

1. **Fixed framing.** Every frame is exactly 960 samples at 48 kHz. A variable
   frame size forces the browser to rebuffer and shows up as a stutter, not a
   latency number.
2. **Generation gating.** Audio belonging to a cancelled generation is dropped at
   this boundary, not merely stopped upstream. Sockets keep delivering after a
   cancel; this is where the promise is kept.
3. **Pacing.** Frames are released on a monotonic deadline so playback does not
   drift when the event loop is busy.
4. **An explicit lifecycle.** A turn begins a generation, streams audio, finishes
   it, and only then can "drained" mean anything. Selection is not delivery: the
   transport acknowledges each frame it accepted, and the drain signal waits for
   that acknowledgement instead of for the queue to look empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .frames import FRAME_SAMPLES, frame_bytes, silent_frame

# The first chunk of a generated utterance is typically pure dead air: Smallest
# Lightning measured 34.6 ms with a peak of 0-6 out of 32768, before the first
# phoneme. Handing that to the egress makes the visitor wait ~50 ms for silence
# and makes `egress` look like framework overhead. Below this threshold the
# samples are inaudible even at full playback gain, so dropping them removes
# nothing anyone can hear.
LEAD_TRIM_THRESHOLD = 96  # ~0.3% of full scale
# Ten milliseconds of pre-roll, not two: a breathy onset ("Hi, I'm Sara") starts
# well below the threshold and climbs over a few milliseconds. The first span
# measured 34.6 ms of padding, so most of the gain survives this protection.
LEAD_TRIM_PREROLL = 480  # samples (10 ms)


def trim_leading_silence(
    pcm: np.ndarray,
    *,
    threshold: int = LEAD_TRIM_THRESHOLD,
    preroll: int = LEAD_TRIM_PREROLL,
) -> np.ndarray:
    """Drop inaudible samples at the start of one utterance chunk.

    An entirely silent chunk becomes empty, which is the honest answer: there is
    nothing to play yet.
    """
    if not pcm.size:
        return pcm
    loud = np.flatnonzero(np.abs(pcm) > threshold)
    if not loud.size:
        return np.zeros(0, dtype=np.int16)
    start = max(0, int(loud[0]) - preroll)
    return pcm[start:] if start else pcm


@dataclass
class AudioChunk:
    pcm: np.ndarray  # int16, 48 kHz mono
    generation: int
    turn_id: str


class EgressFrame(np.ndarray):
    """One paced frame plus the generation and sequence it was selected for.

    A numpy subclass so existing callers can keep treating a frame as samples,
    while the transports can refuse to send a frame whose generation is no
    longer current.
    """

    generation: int
    sequence: int

    def __new__(cls, samples: np.ndarray, generation: int, sequence: int) -> EgressFrame:
        obj = np.asarray(samples).view(cls)
        obj.generation = generation
        obj.sequence = sequence
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.generation = getattr(obj, "generation", 0)
        self.sequence = getattr(obj, "sequence", 0)

    @property
    def silent(self) -> bool:
        return not bool(self.any())


class AudioEgress:
    def __init__(self, *, prebuffer_ms: int = 40, flush_guard_ms: int = 80) -> None:
        self.generation = 0
        self._chunks: deque[AudioChunk] = deque(maxlen=400)
        self._residual = np.zeros(0, dtype=np.int16)
        self._prebuffer_frames = max(1, prebuffer_ms // 20)
        self._flush_guard_until = 0.0
        self._flush_guard_s = flush_guard_ms / 1000.0
        self._deadline: float | None = None
        self._first_audio_sent: set[int] = set()
        # Generations that will produce no more audio. Only these can drain.
        self._finished: set[int] = set()
        # Highest frame sequence emitted per generation, and the highest one the
        # transport confirmed handing off. Drain is decided on the second.
        self._seq: dict[int, int] = {}
        self._last_audio_seq: dict[int, int] = {}
        self._acked: dict[int, int] = {}
        # One drained signal per generation: "every sample this generation will
        # produce has been handed to the transport". The turn lifecycle uses it
        # to stay SPEAKING until the answer is actually over.
        self._drained: dict[int, asyncio.Event] = {}
        self.on_first_audio = None  # optional callback(generation)
        self.underruns = 0
        self.dropped = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ control

    def begin_generation(self, generation: int) -> None:
        """Start a turn. Residual audio from the previous one is discarded.

        Not clearing the residual is how 40 samples of the previous answer ended
        up in front of the next one: the new turn's first frame was built from
        the old remainder plus the new audio.
        """
        self.generation = generation
        self._chunks.clear()
        self._residual = np.zeros(0, dtype=np.int16)
        self._deadline = None
        self._seq[generation] = 0
        self._acked.pop(generation, None)
        self._last_audio_seq.pop(generation, None)
        self._finished.discard(generation)
        self._drained.pop(generation, None)
        # Prune completed generations so the maps do not grow for a long call.
        for stale in [key for key in self._finished if key < generation]:
            self._finished.discard(stale)
            self._drained.pop(stale, None)
            self._seq.pop(stale, None)
            self._last_audio_seq.pop(stale, None)
            self._acked.pop(stale, None)

    # Backwards compatible name; the behaviour is the new lifecycle's.
    def set_generation(self, generation: int) -> None:
        self.begin_generation(generation)

    def finish_generation(self, generation: int) -> None:
        """Declare that no more audio will be queued for this generation.

        Streaming keeps its two-frame pre-buffer, but the end of a turn must
        not: otherwise one remaining frame sits in the queue forever and the
        turn waits out the drain timeout in ``SPEAKING``.
        """
        if generation == self.generation:
            self._finished.add(generation)
            # The legacy "queue looks empty" signal may already have fired
            # before the turn declared itself finished. Re-open the drain and
            # let the transport's acknowledgement close it, so a selected frame
            # is never mistaken for a delivered one.
            event = self._drained.get(generation)
            if event is not None:
                event.clear()
            self._maybe_drained(generation)

    def interrupt(self, generation: int | None = None) -> None:
        """Drop queued audio and emit silence while the browser flushes."""
        self.generation = generation if generation is not None else self.generation + 1
        self._chunks.clear()
        self._residual = np.zeros(0, dtype=np.int16)
        self._flush_guard_until = time.perf_counter() + self._flush_guard_s
        self._deadline = None
        self._finished.discard(self.generation)
        # Release anybody waiting for the old generation to finish playing: it
        # is not going to finish, and the new turn must not wait for it.
        for event in self._drained.values():
            event.set()

    def ack_transport(self, generation: int, sequence: int) -> None:
        """The transport accepted this frame. This is what completes a drain."""
        current = self._acked.get(generation, -1)
        if sequence > current:
            self._acked[generation] = sequence
        self._maybe_drained(generation)

    def _drain_event(self, generation: int) -> asyncio.Event:
        event = self._drained.get(generation)
        if event is None:
            event = asyncio.Event()
            self._drained[generation] = event
        return event

    def _maybe_drained(self, generation: int) -> None:
        if generation != self.generation or generation not in self._finished:
            return
        if self._chunks or self._residual.size:
            return
        if self._acked.get(generation, -1) < self._last_audio_seq.get(generation, -1):
            return
        self._drain_event(generation).set()

    async def wait_transport_drained(self, generation: int, timeout: float = 30.0) -> bool:
        """Wait until this generation has no audio left to hand to the wire.

        This is transport-level completion, not speaker-level: the browser
        still has its own buffer. It is the signal the turn lifecycle needs to
        stop calling itself ``speaking``, and it deliberately releases early on
        an interrupt (the old generation is cancelled, not played out).
        """
        if generation != self.generation:
            return True
        if generation not in self._finished and (
            not self._chunks and len(self._residual) < FRAME_SAMPLES
        ):
            return True
        event = self._drain_event(generation)
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    # Kept as the name the orchestrator already calls.
    async def wait_drained(self, generation: int, timeout: float = 30.0) -> bool:
        return await self.wait_transport_drained(generation, timeout=timeout)

    # -------------------------------------------------------------------- input

    def put(self, pcm: np.ndarray, generation: int, turn_id: str) -> bool:
        if generation != self.generation:
            self.dropped += 1
            return False
        if pcm.size:
            # New audio arrived for this generation: playback is no longer
            # complete.
            self._drain_event(generation).clear()
            if generation not in self._first_audio_sent:
                # Nothing of this generation has been heard yet, so leading
                # silence is dead air in front of the answer, not a pause
                # between sentences. Once audio is playing, the gaps belong to
                # the speech and are left alone.
                pcm = trim_leading_silence(pcm)
            if pcm.size:
                self._chunks.append(AudioChunk(pcm=pcm, generation=generation, turn_id=turn_id))
        return True

    def pending_samples(self) -> int:
        queued = sum(len(chunk.pcm) for chunk in self._chunks)
        return queued + len(self._residual)

    # ------------------------------------------------------------------- output

    def _next_chunk_frame(self, *, flush: bool) -> np.ndarray:
        while len(self._residual) < FRAME_SAMPLES and self._chunks:
            chunk = self._chunks.popleft()
            if chunk.generation != self.generation:
                self.dropped += 1
                continue
            if self._residual.size:
                self._residual = np.concatenate([self._residual, chunk.pcm])
            else:
                self._residual = chunk.pcm
        if len(self._residual) >= FRAME_SAMPLES:
            frame = self._residual[:FRAME_SAMPLES]
            self._residual = self._residual[FRAME_SAMPLES:]
            return frame
        if flush and self._residual.size:
            # The end of a turn: a partial frame is padded with silence and
            # released now, not held for a next chunk that will never come.
            frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
            frame[: self._residual.size] = self._residual
            self._residual = np.zeros(0, dtype=np.int16)
            return frame
        return silent_frame(FRAME_SAMPLES)

    def _sequence(self, generation: int) -> int:
        sequence = self._seq.get(generation, 0)
        self._seq[generation] = sequence + 1
        return sequence

    async def next_frame(self) -> EgressFrame:
        """Return one paced 20 ms frame, tagged with its generation/sequence."""
        async with self._lock:
            if time.perf_counter() < self._flush_guard_until:
                generation = self.generation
                return EgressFrame(silent_frame(FRAME_SAMPLES), generation, self._sequence(generation))

            generation = self.generation
            sequence = self._sequence(generation)
            first_for_generation = generation not in self._first_audio_sent
            finished = generation in self._finished
            # Streaming keeps a pre-buffer; the first frame and the end of a
            # turn do not. The first frame is what the visitor is waiting for;
            # the last one must be flushed rather than stranded.
            needed = 0 if finished else (
                FRAME_SAMPLES if first_for_generation else FRAME_SAMPLES * self._prebuffer_frames
            )
            pending = self.pending_samples()
            has_residual = bool(self._residual.size)
            enough = pending >= needed if needed else (bool(self._chunks) or has_residual)
            if finished and not self._chunks and not has_residual:
                frame = silent_frame(FRAME_SAMPLES)
            elif not enough and not has_residual:
                # An underrun means the visitor heard playback stop mid-word.
                # Before this generation has played anything there is nothing to
                # stutter: the sender is waiting for the first voiced samples,
                # and that wait is already measured as `egress`. Counting it here
                # made every turn report a stutter it did not have, which is how
                # a real one would hide.
                if generation in self._first_audio_sent:
                    self.underruns += 1
                frame = silent_frame(FRAME_SAMPLES)
            else:
                frame = self._next_chunk_frame(flush=finished)
                if frame.any() and first_for_generation:
                    self._first_audio_sent.add(generation)
                    if self.on_first_audio is not None:
                        with contextlib.suppress(Exception):
                            self.on_first_audio(generation)

            if frame.any():
                self._last_audio_seq[generation] = sequence

            now = time.perf_counter()
            if finished:
                # Only an acknowledged last frame completes the drain.
                self._maybe_drained(generation)
            elif (
                generation in self._first_audio_sent
                and not self._chunks
                and len(self._residual) < FRAME_SAMPLES
            ):
                # Legacy signal for callers that never call finish_generation
                # (tests, short notices). Real turns take the ack path above.
                self._drain_event(generation).set()

            if self._deadline is None:
                # Nothing has been paced yet, so there is no previous frame to
                # keep a 20 ms cadence from. Send immediately and start the
                # clock here; otherwise every turn pays one frame of silence.
                self._deadline = now
                delay = 0.0
            else:
                self._deadline += 0.020
                delay = self._deadline - now
                if delay < -0.100:
                    # We fell more than five frames behind: resynchronise instead
                    # of trying to catch up by bursting frames.
                    self._deadline = now
                    delay = 0.0
        if delay > 0:
            await asyncio.sleep(delay)
        if self.generation != generation:
            # An interrupt landed while this frame was pacing. It belongs to a
            # generation the visitor must never hear again.
            current = self.generation
            return EgressFrame(silent_frame(FRAME_SAMPLES), current, self._sequence(current))
        return EgressFrame(frame, generation, sequence)

    def drain_snapshot(self) -> dict:
        return {
            "generation": self.generation,
            "pending_ms": round(self.pending_samples() / 48.0, 1),
            "underruns": self.underruns,
            "dropped_chunks": self.dropped,
        }


def frame_payload(samples: np.ndarray) -> bytes:
    return frame_bytes(samples)
