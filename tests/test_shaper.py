"""Egress shaper: fixed framing, generation gating, pacing."""

from __future__ import annotations

import numpy as np
from sarang.audio.frames import FRAME_SAMPLES
from sarang.audio.shaper import AudioEgress


def pcm(count: int, value: int = 1_000) -> np.ndarray:
    return np.full(count, value, dtype=np.int16)


async def test_frames_are_always_exactly_960_samples():
    egress = AudioEgress()
    egress.set_generation(1)
    egress.put(pcm(3_000), 1, "trn_1")
    for _ in range(6):
        frame = await egress.next_frame()
        assert frame.dtype == np.int16
        assert frame.size == FRAME_SAMPLES


async def test_short_tail_is_padded_with_silence():
    egress = AudioEgress()
    egress.set_generation(1)
    egress.put(pcm(2_500), 1, "trn_1")
    first = await egress.next_frame()
    second = await egress.next_frame()
    assert first.size == FRAME_SAMPLES and first.any()
    assert second.size == FRAME_SAMPLES and second.any()
    # 580 samples remain; a partial frame is held rather than emitted short,
    # and the next frame is silence of the correct size.
    tail = await egress.next_frame()
    assert tail.size == FRAME_SAMPLES
    assert not tail.any()
    assert egress.pending_samples() == 580


async def test_stale_generation_is_rejected_at_put():
    egress = AudioEgress()
    egress.set_generation(3)
    assert egress.put(pcm(960), 2, "old") is False
    assert egress.dropped == 1
    assert egress.pending_samples() == 0


async def test_generation_bump_drops_audio_already_queued():
    egress = AudioEgress()
    egress.set_generation(1)
    egress.put(pcm(4_000), 1, "trn_1")
    egress.interrupt(2)
    assert egress.pending_samples() == 0
    frame = await egress.next_frame()
    assert not frame.any()


async def test_interrupt_guard_emits_silence_then_accepts_new_audio():
    egress = AudioEgress(flush_guard_ms=10)
    egress.set_generation(1)
    egress.put(pcm(1_920), 1, "trn_1")
    egress.interrupt(2)
    guard = await egress.next_frame()
    assert not guard.any()
    egress.put(pcm(1_920, value=2_000), 2, "trn_2")
    await _wait_guard()
    frame = await egress.next_frame()
    assert frame.any()


async def _wait_guard() -> None:
    import asyncio

    await asyncio.sleep(0.02)


async def test_first_audio_callback_fires_once_per_generation():
    seen: list[int] = []
    egress = AudioEgress()
    egress.on_first_audio = seen.append
    egress.set_generation(1)
    egress.put(pcm(4_000), 1, "trn_1")
    for _ in range(4):
        await egress.next_frame()
    assert seen == [1]
    egress.set_generation(2)
    egress.put(pcm(4_000), 2, "trn_2")
    for _ in range(4):
        await egress.next_frame()
    assert seen == [1, 2]


async def test_first_frame_of_a_generation_is_not_paced():
    """The wait the visitor feels is this frame; it must not pay a cadence."""
    import time

    egress = AudioEgress()
    egress.set_generation(1)
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")

    started = time.perf_counter()
    frame = await egress.next_frame()
    first_ms = (time.perf_counter() - started) * 1000

    assert frame.any()
    assert first_ms < 10.0, f"first frame held for {first_ms:.1f} ms"

    # Subsequent frames do keep the 20 ms cadence.
    egress.put(pcm(FRAME_SAMPLES * 4), 1, "trn_1")
    started = time.perf_counter()
    await egress.next_frame()
    assert (time.perf_counter() - started) * 1000 >= 15.0


async def test_first_frame_needs_only_one_frame_of_audio():
    egress = AudioEgress(prebuffer_ms=40)
    egress.set_generation(1)
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")

    frame = await egress.next_frame()

    assert frame.any()
    assert egress.underruns == 0


async def test_after_the_first_frame_one_frame_is_not_enough_to_pace():
    """The pre-buffer still governs resumption, but never the first frame."""
    egress = AudioEgress(prebuffer_ms=40)
    egress.set_generation(1)

    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    assert (await egress.next_frame()).any()

    # A single new frame is below the two-frame pre-buffer, so the shaper would
    # rather emit silence once than start a stutter it cannot finish.
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    assert not (await egress.next_frame()).any()
    assert egress.underruns == 1


async def test_underrun_is_counted_and_silent():
    egress = AudioEgress()
    egress.set_generation(1)
    frame = await egress.next_frame()
    assert not frame.any()
    assert egress.underruns == 1


async def test_drain_snapshot_reports_pending_ms():
    egress = AudioEgress()
    egress.set_generation(1)
    egress.put(pcm(9_600), 1, "trn_1")
    snapshot = egress.drain_snapshot()
    assert snapshot["pending_ms"] == 200.0
    assert snapshot["generation"] == 1
