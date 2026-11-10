"""The agent as Sanket's voice agent: fit, motivation, community and scope.

Two product rules are pinned here:

* evaluative questions are routed to curated profile/role-fit blocks, not to a
  random project;
* a question unrelated to Sanket gets the polite apology path, while a question
  *about* him that is undocumented stays on the not-documented path.
"""

from __future__ import annotations

from sarang.kb.loader import load_kb
from sarang.kb.lookup import lookup
from sarang.kb.render import MAX_CONTEXT_CHARS, render_context
from sarang.llm.local import OFF_TOPIC_FALLBACK, answer_from
from sarang.llm.prompt import SYSTEM_PROMPT


def test_fit_question_routes_to_capabilities_and_role_fit():
    result = lookup("Is Sanket a good fit for an Agentic AI Engineer role?")
    assert result.confident
    assert result.reason == "role fit"
    kinds = [block.kind for block in result.blocks]
    assert kinds[0] == "capabilities"
    assert "role_fit" in kinds


def test_fit_context_stays_inside_the_prompt_budget():
    result = lookup("Would he be a good match for this role?")
    context = render_context(result.blocks)
    assert len(context) <= MAX_CONTEXT_CHARS
    assert "Open to roles" in context


def test_motivation_question_routes_to_profile():
    result = lookup("What motivates Sanket?")
    assert result.confident
    assert result.reason == "profile"
    assert result.top is not None
    assert result.top.id == "profile"


def test_vision_mission_and_values_route_to_profile():
    for question in (
        "What is his vision?",
        "What is his mission right now?",
        "What are his values?",
    ):
        result = lookup(question)
        assert result.top is not None, question
        assert result.top.id == "profile", question


def test_community_and_hobbies_route_to_profile():
    for question, expected in (
        ("What community work does he do?", "profile"),
        ("What are his hobbies?", "interests"),
        ("Does he play chess?", "interests"),
    ):
        result = lookup(question)
        assert result.top is not None, question
        assert result.top.id == expected, question


def test_unrelated_questions_get_the_apology_block():
    for question in (
        "Tell me a joke",
        "What is the weather in Pune?",
        "Who is the president of India?",
        "Write me a Python script to scrape a site",
    ):
        result = lookup(question)
        assert result.confident, question
        assert result.reason in {"off topic", "task request"}, question
        assert result.top is not None
        assert result.top.kind == "off_topic", question


def test_question_about_him_is_not_treated_as_off_topic():
    # Undocumented, but about Sanket: the not-documented path, not the apology.
    result = lookup("Did he play cricket?")
    assert result.reason != "off topic"
    assert not result.confident


def test_fit_blocks_do_not_leak_into_ordinary_answers():
    result = lookup("What did he do at ARAI?")
    assert result.top is not None
    assert result.top.id == "experience:arai"
    kinds = {block.kind for block in result.blocks}
    assert "capabilities" not in kinds
    assert "role_fit" not in kinds


def test_prompt_names_him_as_the_subject_not_the_portfolio():
    assert "Sanket Sonwane's voice agent" in SYSTEM_PROMPT
    lowered = SYSTEM_PROMPT.lower()
    assert "give a view" in lowered
    assert "apologise" in lowered
    assert "promise an outcome" in lowered


def test_capabilities_block_is_bounded():
    kb = load_kb()
    block = kb.by_id["capabilities"]
    assert 300 <= len(block.text) <= 1_000


def test_local_fit_answer_is_positive_and_grounded():
    question = "Is he a good fit for an LLM engineer role?"
    text = answer_from(question, lookup(question))
    assert "production" in text.lower()
    assert "Happy to go deeper" in text
    # Never claim an outcome or a score.
    assert "guarantee" not in text.lower()
    assert "%" not in text


def test_local_profile_answer_uses_the_curated_field():
    vision = answer_from("What is his vision?", lookup("What is his vision?"))
    assert "trust" in vision.lower()
    community = answer_from(
        "What community work does he do?", lookup("What community work does he do?")
    )
    assert "Developer Tech Society" in community


def test_local_off_topic_answer_is_the_humble_line():
    text = answer_from("Tell me a joke", lookup("Tell me a joke"))
    assert text == OFF_TOPIC_FALLBACK
    assert text.startswith("Sorry")
