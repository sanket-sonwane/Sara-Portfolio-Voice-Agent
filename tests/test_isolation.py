"""Session isolation and the three text states of a turn."""

from __future__ import annotations

import asyncio

import numpy as np
from sarang.audio.frames import FRAME_SAMPLES
from sarang.audio.shaper import AudioEgress
from sarang.bus import Bus
from sarang.schemas import Envelope, EventType
from sarang.text_state import TurnText


def pcm(count: int, value: int = 1_000) -> np.ndarray:
    return np.full(count, value, dtype=np.int16)


# ------------------------------------------------------------------- the bus


def test_publish_reaches_only_the_named_session():
    bus = Bus()
    _, queue_a = bus.subscribe("ses_a")
    _, queue_b = bus.subscribe("ses_b")

    bus.publish(Envelope(type=EventType.STATE, session_id="ses_a", payload={"state": "idle"}))

    assert queue_a.qsize() == 1
    assert queue_b.qsize() == 0


def test_global_subscriber_still_sees_everything():
    bus = Bus()
    _, global_queue = bus.subscribe()
    _, queue_a = bus.subscribe("ses_a")

    bus.publish(Envelope(type=EventType.STATE, session_id="ses_a"))
    bus.publish(Envelope(type=EventType.STATE, session_id="ses_b"))

    assert global_queue.qsize() == 2
    assert queue_a.qsize() == 1


def test_unsubscribe_removes_the_session_mapping():
    bus = Bus()
    key, queue = bus.subscribe("ses_a")
    bus.unsubscribe(key)

    bus.publish(Envelope(type=EventType.STATE, session_id="ses_a"))
    assert queue.qsize() == 0
    assert bus._by_session == {}

# ----------------------------------------------------------------- the egress


async def test_finish_releases_the_last_single_frame():
    """The two-frame pre-buffer must not strand the final 20 ms of an answer."""
    egress = AudioEgress(prebuffer_ms=40)
    egress.begin_generation(1)
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    assert (await egress.next_frame()).any()

    # One more complete frame, then the turn ends. Streaming would hold it for
    # the pre-buffer forever; the flush releases it.
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    egress.finish_generation(1)
    frame = await egress.next_frame()

    assert frame.any()
    assert egress.pending_samples() == 0


async def test_finish_flushes_a_sub_frame_tail():
    egress = AudioEgress()
    egress.begin_generation(1)
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    await egress.next_frame()
    egress.put(pcm(100), 1, "trn_1")
    egress.finish_generation(1)

    tail = await egress.next_frame()

    assert tail.size == FRAME_SAMPLES
    assert tail[:100].any()
    assert not tail[100:].any()
    assert egress.pending_samples() == 0


async def test_begin_generation_discards_the_previous_residual():
    egress = AudioEgress()
    egress.begin_generation(1)
    egress.put(pcm(40), 1, "trn_1")

    egress.begin_generation(2)
    egress.put(pcm(FRAME_SAMPLES), 2, "trn_2")
    frame = await egress.next_frame()

    assert np.array_equal(np.asarray(frame), pcm(FRAME_SAMPLES))


async def test_interrupt_during_pacing_never_releases_the_old_frame():
    egress = AudioEgress(flush_guard_ms=0)
    egress.begin_generation(1)
    egress.put(pcm(FRAME_SAMPLES * 4), 1, "trn_1")
    assert (await egress.next_frame()).any()

    # The next frame is selected, then sleeps for the 20 ms cadence. An
    # interrupt lands during that sleep; the frame it already chose must not be
    # returned as if it still belonged to the current generation.
    paced = asyncio.create_task(egress.next_frame())
    await asyncio.sleep(0.005)
    egress.interrupt(2)
    frame = await paced

    assert frame.generation == 2
    assert not frame.any()


async def test_drain_waits_for_the_transport_acknowledgement():
    egress = AudioEgress()
    egress.begin_generation(1)
    egress.put(pcm(FRAME_SAMPLES), 1, "trn_1")
    frame = await egress.next_frame()
    egress.finish_generation(1)

    waiter = asyncio.create_task(egress.wait_transport_drained(1, timeout=1.0))
    await asyncio.sleep(0)
    assert not waiter.done()

    egress.ack_transport(1, int(frame.sequence))

    assert await asyncio.wait_for(waiter, timeout=1.0) is True


# -------------------------------------------------------------- text states


def test_generated_and_committed_are_not_the_same_answer():
    text = TurnText(generated="One. Two. Three.")
    text.commit("One.")
    text.commit("Two.")

    assert text.generated == "One. Two. Three."
    assert text.committed == "One. Two."
    # The transcript claims only what was actually spoken.
    assert text.transcript() == "One. Two."


def test_a_generated_but_never_committed_answer_publishes_nothing_spoken():
    text = TurnText(generated="An answer the provider never synthesised.")
    assert text.committed == ""
    # Falling back to generated is the only honest option when nothing was
    # spoken at all, and the payload still distinguishes the two.
    assert text.transcript() == "An answer the provider never synthesised."
    assert text.payload()["committed_text"] == ""


def test_audible_is_a_prefix_reported_by_the_browser():
    text = TurnText(generated="One. Two. Three.")
    text.commit("One.")
    text.commit("Two.")

    text.note_audible(chars=5)
    assert text.audible == "One. "
    assert text.payload()["audible_text"] == "One. "

    # A report that is not a prefix of the committed text cannot invent speech.
    text.note_audible(text="Something else entirely")
    assert text.audible == "One. "

    text.note_audible(text="One. Two.")
    assert text.audible == "One. Two."
