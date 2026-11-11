"""Speculation: gates before an arm is ever replayed.

The rule these pin: speculation may never change what the visitor hears. An
arm is buffered while the visitor talks, never spoken pre-confirm, and
replayed only when it was answering the same question, with the same
evidence, under the same prompt. Everything else must fall back to a fresh
stream - the latency win is optional, correctness is not.
"""

from __future__ import annotations

import asyncio

from sarang.llm.base import Done, TextDelta
from sarang.speculative import Speculator

PARTIAL = "what projects did he build"


class FakeLLM:
    """Records stream() calls and yields one canned answer."""

    def __init__(self, answer: str = "Sanket built TattvaQuest, his commerce gateway.",
                 *, fail_first: bool = False) -> None:
        self.calls = 0
        self.answer = answer
        self.fail_first = fail_first

    async def stream(self, messages, tools=None, max_tokens=0, temperature=0.0):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("provider refused the connection")
        yield TextDelta(text=self.answer)
        yield Done(usage={})

    async def close(self) -> None:
        return None


async def _arm(speculator: Speculator, partial: str):
    """Note a partial twice (the stability requirement) and let the arm run."""
    await speculator.note(partial)
    candidate = await speculator.note(partial)
    assert candidate is not None
    # One tick to start the armer, one for the pump to drain the stream.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return candidate


# ------------------------------------------------------------------ gates


async def test_partial_of_a_different_question_is_not_adopted():
    speculator = Speculator()
    candidate = await _arm(speculator, "tell me about tattvaquest")
    assert candidate is not None

    adopted = speculator.consume("what did he build at arai")

    assert adopted is None
    assert speculator.adopted == 0
    assert speculator.wasted == 1


async def test_boundary_question_is_not_adopted_over_the_same_topic():
    """The evidence gate, not the token gate, must refuse this one.

    The final transcript keeps two of the hypothesis's three topical tokens
    (0.67 >= 0.6), so token overlap alone would adopt an arm grounded on the
    ARAI internship for a salary question that routes to the boundaries
    block.
    """
    speculator = Speculator()
    candidate = await _arm(speculator, "what did he build at arai")
    assert candidate is not None
    assert not candidate.result.blocked

    adopted = speculator.consume("what did he build at arai and also his salary")

    assert adopted is None
    assert speculator.wasted == 1


async def test_matching_partial_is_adopted():
    speculator = Speculator()
    candidate = await _arm(speculator, PARTIAL)

    adopted = speculator.consume(PARTIAL)

    assert adopted is candidate
    assert adopted is not None and adopted.used
    assert speculator.adopted == 1
    assert speculator.wasted == 0


async def test_reset_cancels_a_running_arm_and_counts_the_waste():
    cancelled = False

    async def armer(spec) -> bool:
        nonlocal cancelled
        spec.arm_queue = asyncio.Queue()
        try:
            await asyncio.Event().wait()  # an arm that would stream forever
        except asyncio.CancelledError:
            cancelled = True
            raise
        return True

    speculator = Speculator(armer=armer)
    candidate = await _arm(speculator, PARTIAL)
    assert candidate.arm_queue is not None
    task = candidate.arm_task
    assert task is not None and not task.done()

    speculator.reset()

    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert cancelled
    assert speculator.arm_wasted == 1
    # The next note starts from a clean slate.
    assert await speculator.note(PARTIAL) is None


# ------------------------------------------------------- replay end to end


async def test_arm_is_replayed_instead_of_a_fresh_stream(orchestrator, session):
    fake = FakeLLM()
    orchestrator.llm = fake
    speculator = session.speculation

    candidate = await _arm(speculator, PARTIAL)
    assert fake.calls == 1
    assert candidate.arm_error is None
    assert speculator.stats()["armed"] == 1

    await orchestrator.run_turn(session, PARTIAL)

    # No second provider stream: the answer was already in the queue.
    assert fake.calls == 1
    assert speculator.arm_adopted == 1
    assert session.history[-1].role == "assistant"
    assert "TattvaQuest" in session.history[-1].content


async def test_arm_is_discarded_when_the_history_changed(orchestrator, session):
    """A turn that finished mid-utterance adds to the history after the arm."""
    fake = FakeLLM()
    orchestrator.llm = fake
    speculator = session.speculation

    await _arm(speculator, PARTIAL)
    assert fake.calls == 1

    session.remember("assistant", "Something said while the visitor talked.")

    await orchestrator.run_turn(session, PARTIAL)

    # The stale arm was refused (prompt inequality) and a fresh stream ran.
    assert fake.calls == 2
    assert speculator.arm_adopted == 0
    assert speculator.arm_wasted == 1
    # The lookup-level speculation is independent and still counts as adopted.
    assert speculator.adopted == 1


async def test_arm_that_failed_before_the_first_word_streams_fresh(orchestrator, session):
    fake = FakeLLM(fail_first=True)
    orchestrator.llm = fake
    speculator = session.speculation

    candidate = await _arm(speculator, PARTIAL)
    assert fake.calls == 1
    assert candidate.arm_error is not None

    await orchestrator.run_turn(session, PARTIAL)

    assert fake.calls == 2
    assert speculator.arm_wasted == 1
    assert speculator.arm_adopted == 0
    assert session.history[-1].role == "assistant"
    assert "TattvaQuest" in session.history[-1].content


async def test_no_llm_means_no_arm_but_lookup_still_warms(orchestrator, session):
    orchestrator.llm = None  # the local profile: build_llm returns None
    speculator = session.speculation

    candidate = await _arm(speculator, PARTIAL)

    assert candidate is not None
    assert candidate.arm_queue is None
    assert speculator.stats()["armed"] == 0
    # The warm lookup itself still works and is adoptable.
    adopted = speculator.consume(PARTIAL)
    assert adopted is candidate
