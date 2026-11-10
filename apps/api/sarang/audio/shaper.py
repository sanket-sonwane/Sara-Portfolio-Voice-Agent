"""Egress shaper.

The only writer to the outbound audio track. It guarantees three properties:

1. **Fixed framing.** Every frame is exactly 960 samples at 48 kHz. A variable
   frame size forces the browser to rebuffer and shows up as a stutter, not a
   latency number.
2. **Generation gating.** Audio belonging to a cancelled generation is dropped at
   this boundary, not merely stopped upstream. Sockets keep delivering after a
   cancel; this is where the promise is kept.
3. **Pacing.** Frames are released on a monotonic deadline so playback does not
   drift when the event loop is busy.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .frames import FRAME_SAMPLES, frame_bytes, silent_frame


@dataclass
class AudioChunk:
    pcm: np.ndarray  # int16, 48 kHz mono
    generation: int
    turn_id: str


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
        self.on_first_audio = None  # optional callback(generation)
        self.underruns = 0
        self.dropped = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ control

    def set_generation(self, generation: int) -> None:
        self.generation = generation

    def interrupt(self, generation: int | None = None) -> None:
        """Drop queued audio and emit silence while the browser flushes."""
        self.generation = generation if generation is not None else self.generation + 1
        self._chunks.clear()
        self._residual = np.zeros(0, dtype=np.int16)
        self._flush_guard_until = time.perf_counter() + self._flush_guard_s
        self._deadline = None

    # -------------------------------------------------------------------- input

    def put(self, pcm: np.ndarray, generation: int, turn_id: str) -> bool:
        if generation != self.generation:
            self.dropped += 1
            return False
        if pcm.size:
            self._chunks.append(AudioChunk(pcm=pcm, generation=generation, turn_id=turn_id))
        return True

    def pending_samples(self) -> int:
        queued = sum(len(chunk.pcm) for chunk in self._chunks)
        return queued + len(self._residual)

    # ------------------------------------------------------------------- output

    def _next_chunk_frame(self) -> np.ndarray:
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
        return silent_frame(FRAME_SAMPLES)

    async def next_frame(self) -> np.ndarray:
        """Return one 20 ms frame, paced against a monotonic deadline."""
        async with self._lock:
            if time.perf_counter() < self._flush_guard_until:
                return silent_frame(FRAME_SAMPLES)

            generation = self.generation
            first_for_generation = generation not in self._first_audio_sent
            # The first frame of a turn is the one the visitor is waiting for.
            # Holding it for the 40 ms pre-buffer costs latency and buys nothing
            # on a fresh generation, because a single 20 ms frame is already
            # playable. Later frames keep the pre-buffer.
            needed = FRAME_SAMPLES if first_for_generation else FRAME_SAMPLES * self._prebuffer_frames
            enough = self.pending_samples() >= needed
            if not enough and not self._residual.size:
                self.underruns += 1
                frame = silent_frame(FRAME_SAMPLES)
            else:
                frame = self._next_chunk_frame()
                if frame.any() and first_for_generation:
                    self._first_audio_sent.add(generation)
                    if self.on_first_audio is not None:
                        with contextlib.suppress(Exception):
                            self.on_first_audio(generation)

            now = time.perf_counter()
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
        return frame

    def drain_snapshot(self) -> dict:
        return {
            "generation": self.generation,
            "pending_ms": round(self.pending_samples() / 48.0, 1),
            "underruns": self.underruns,
            "dropped_chunks": self.dropped,
        }


def frame_payload(samples: np.ndarray) -> bytes:
    return frame_bytes(samples)
