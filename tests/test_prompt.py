"""Prompt budget: behaviour in the system prompt, facts injected per turn."""

from __future__ import annotations

from sarang.llm.base import Message
from sarang.llm.prompt import SYSTEM_PROMPT, build_messages, build_system


def test_system_prompt_stays_inside_budget():
    assert len(SYSTEM_PROMPT) <= 1_250


def test_system_prompt_does_not_carry_a_portfolio_digest():
    for term in ("TattvaQuest", "Class-Node", "9.14", "ARAI", "pgvector"):
        assert term not in SYSTEM_PROMPT


def test_system_prompt_states_the_grounding_rules():
    lowered = SYSTEM_PROMPT.lower()
    assert "only from the facts" in lowered
    assert "don't have that detail" in lowered
    assert "submitted" in lowered
    assert "test mode" in lowered
    # The salary rule must be scoped, or the model recites it on every turn.
    assert "only if the visitor asks" in lowered


def test_system_prompt_forbids_filler_openers():
    """The model must not wrap answers in 'However,' / 'I mean' discourse.

    The visitor complained about exactly this: answers that open with the
    same filler words the recogniser transcribes from speech.
    """
    lowered = SYSTEM_PROMPT.lower()
    assert "never open with filler" in lowered
    assert "'i mean'" in lowered


def test_lexicon_is_only_added_within_budget():
    base = build_system()
    assert base == SYSTEM_PROMPT
    with_lexicon = build_system(["TattvaQuest", "QA-Forge"] * 40)
    assert len(with_lexicon) <= 1_250


def test_build_messages_puts_facts_in_the_system_message():
    messages = build_messages(
        "Tell me about QA-Forge",
        "[project:qa-forge] QA-Forge is a multi-agent auditing platform.",
        history=[Message(role="user", content="Hi"), Message(role="assistant", content="Hello")],
    )
    assert messages[0].role == "system"
    assert "qa-forge" in messages[0].content.lower()
    assert messages[-1].role == "user"
    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]


def test_context_is_optional_and_still_valid_prompt():
    messages = build_messages("Who is Sanket?", "(no matching portfolio evidence)")
    assert "no matching portfolio evidence" in messages[0].content
