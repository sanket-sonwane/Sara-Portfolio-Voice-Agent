"""Epoch discipline: a barge-in invalidates everything from the old generation."""

from __future__ import annotations

import asyncio

import numpy as np
from sarang.schemas import SessionState


async def test_interrupt_bumps_generation_and_clears_audio(orchestrator, session):
    session.generation = 1
    session.egress.set_generation(1)
    session.egress.put(np.full(4_800, 500, dtype=np.int16), 1, "trn_1")
    assert session.egress.pending_samples() > 0

    await orchestrator.interrupt(session, "barge-in")

    assert session.generation == 2
    assert session.interruptions == 1
    assert session.egress.pending_samples() == 0
    assert session.state == SessionState.INTERRUPTED


async def test_interrupt_cancels_the_active_turn_task(orchestrator, session):
    task = asyncio.create_task(asyncio.sleep(5))
    session.turn_task = task
    await orchestrator.interrupt(session)
    await asyncio.sleep(0)
    assert task.cancelled() or task.cancelling()


async def test_stale_audio_cannot_be_queued_after_interrupt(orchestrator, session):
    await orchestrator.interrupt(session)
    generation = session.generation
    assert session.egress.put(np.ones(960, dtype=np.int16), generation - 1, "old") is False


async def test_interrupted_state_returns_to_listening(orchestrator, session):
    await orchestrator.interrupt(session)
    await asyncio.sleep(0.2)
    assert session.state == SessionState.LISTENING


async def test_interrupt_publishes_epoch_with_new_generation(orchestrator, session):
    bus = orchestrator.registry.bus
    key, queue = bus.subscribe()
    await orchestrator.interrupt(session)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    epochs = [event for event in events if event.type.value == "epoch"]
    assert len(epochs) == 1
    assert epochs[0].payload["generation"] == session.generation
    assert epochs[0].payload["reason"] == "barge-in"


async def test_cancelled_turn_marks_its_metrics(orchestrator, session):
    class SlowTTS:
        name = "slow"

        async def synthesize(self, text, *, context_id, final=True, metadata=None):
            await asyncio.sleep(5)
            if False:  # pragma: no cover - keeps this an async generator
                yield None

        async def close(self):
            return None

    orchestrator.tts = SlowTTS()
    task = asyncio.create_task(orchestrator.run_turn(session, "Who is Sanket?"))
    session.turn_task = task
    await asyncio.sleep(0.15)
    generation = session.generation
    await orchestrator.interrupt(session)
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    marks = session.marks[generation]
    assert marks.cancelled
