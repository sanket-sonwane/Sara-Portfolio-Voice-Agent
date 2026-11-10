"""End-to-end turn engine with the zero-key local pipeline."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sarang.orchestrator import filler_voice_tag
from sarang.schemas import Envelope


def test_filler_cache_is_keyed_by_provider_and_voice():
    """Switching TTS arm must not reuse the previous voice's cached fillers."""
    smallest = SimpleNamespace(
        name="smallest", model="lightning_v3.1_pro", voice_id="rhea", language="hi"
    )
    cartesia = SimpleNamespace(
        name="cartesia", model="sonic-latest", voice_id="ella", language="en"
    )

    assert filler_voice_tag(smallest) == "smallest-lightning_v3.1_pro-rhea-hi"
    assert filler_voice_tag(cartesia) == "cartesia-sonic-latest-ella-en"


def test_filler_cache_tag_uses_the_fallback_primary():
    fallback = SimpleNamespace(
        name="smallest",
        primary=SimpleNamespace(model="m", voice_id="rhea", language="hi"),
    )

    assert filler_voice_tag(fallback) == "smallest-m-rhea-hi"


async def collect(queue: asyncio.Queue[Envelope], wanted: str, timeout: float = 8.0) -> list[Envelope]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    seen: list[Envelope] = []
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            kinds = [event.type.value for event in seen]
            raise AssertionError(f"timed out waiting for {wanted}; saw {kinds}")
        event = await asyncio.wait_for(queue.get(), timeout=remaining)
        seen.append(event)
        if event.type.value == wanted:
            return seen


async def test_grounded_turn_produces_answer_and_metrics(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "What did he do at ARAI?")
    events = await collect(queue, "metrics.turn")

    final = [event for event in events if event.type.value == "assistant.final"]
    metrics = [event for event in events if event.type.value == "metrics.turn"][0]
    assert final
    text = final[-1].payload["text"]
    assert "ARAI" in text
    # The spoken cap may drop the tail metric segment; grounding is proven by the
    # first documented metric surviving, not by a specific tail value.
    assert "10 hours" in text or "99.94" in text
    assert metrics.payload["source"] == "text"
    assert metrics.payload["cancelled"] is False
    assert metrics.payload["answer_first_audio_ms"] > 0
    assert "lookup" in metrics.payload["durations_ms"]
    assert final[-1].turn_id == metrics.turn_id


async def test_currency_question_is_refused_not_guessed(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "What is his salary expectation?")
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert "compensation" in text.lower()
    assert "₹" not in text and "lakh" not in text.lower()


async def test_pending_project_is_refused(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "Tell me about the CPU scheduling project")
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert "documented" in text.lower()


async def test_history_is_kept_for_follow_ups(orchestrator, session):
    await orchestrator.run_turn(session, "What did he do at ARAI?")
    await orchestrator.run_turn(session, "Tell me more about that internship.")
    roles = [message.role for message in session.history]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2
    assert len(session.history) <= 8


async def test_final_answer_is_published_before_metrics(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "What is TattvaQuest?")
    events = await collect(queue, "metrics.turn")
    kinds = [event.type.value for event in events]
    assert kinds.index("assistant.final") < kinds.index("metrics.turn")
    assert "thinking" in [event.payload.get("state") for event in events if event.type.value == "state"]


async def test_spoken_answer_is_hard_capped_at_a_sentence_boundary(orchestrator, session):
    # The prompt asks for 35 words; the engine must not rely on the model obeying.
    orchestrator.settings.max_spoken_words = 25
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "What did he do at ARAI?")
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert len(text.split()) <= 25
    assert "99.94" not in text  # the tail is not spoken


async def test_follow_up_keeps_the_previous_evidence(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, "Which internship was about network security?")
    await collect(queue, "assistant.final")
    await orchestrator.run_turn(session, "What were the numbers?")
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert "98" in text
    assert "EPIC" not in text


async def test_greeting_is_spoken_once(orchestrator, session):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.greet(session)
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert text.startswith("Hi, I'm Sara")
    await orchestrator.greet(session)
    while not queue.empty():
        assert queue.get_nowait().type.value != "assistant.final"


async def test_empty_text_does_not_start_a_turn(orchestrator, session):
    before = session.turn_counter
    await orchestrator.run_turn(session, "   ")
    assert session.turn_counter == before


async def test_generation_increases_monotonically(orchestrator, session):
    await orchestrator.run_turn(session, "Who is Sanket?")
    first = session.generation
    await orchestrator.run_turn(session, "What are his skills?")
    assert session.generation == first + 1


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about QA-Forge",
        "Explain the municipal copilot's hybrid retrieval",
        "What was the result of the AEUE paper?",
        "How many weekly users did Class-Node have?",
        "Did he work at NVIDIA?",
    ],
)
async def test_question_set_never_returns_empty_answers(orchestrator, session, question):
    key, queue = orchestrator.registry.bus.subscribe()
    await orchestrator.run_turn(session, question)
    events = await collect(queue, "assistant.final")
    text = [event for event in events if event.type.value == "assistant.final"][0].payload["text"]
    assert len(text.strip()) > 10
