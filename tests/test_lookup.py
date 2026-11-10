"""Tier-1 lookup: routing, confidence and the refusal path."""

from __future__ import annotations

from sarang.kb.lookup import lookup, tokenize


def test_cgpa_routes_to_education():
    result = lookup("What is Sanket's CGPA?")
    assert result.top is not None
    assert result.top.kind == "education"
    assert result.confident


def test_named_project_routes_to_that_project():
    result = lookup("Tell me about TattvaQuest")
    assert result.top is not None
    assert result.top.id == "project:tattvaquest"
    assert result.confident


def test_named_internship_routes_to_experience():
    result = lookup("What did he do at ARAI?")
    assert result.top is not None
    assert result.top.kind == "experience"
    assert "arai" in result.top.id


def test_owner_name_does_not_outrank_the_topic_of_the_question():
    result = lookup("What did Sanket build during the ARAI internship?")
    assert result.top is not None
    assert result.top.id == "experience:arai"
    assert result.confident


def test_owner_name_only_generic_project_question_returns_the_overview():
    result = lookup("What did Sanket build?")
    assert result.reason == "generic project overview"
    assert result.top is not None
    assert result.top.kind == "project"


def test_salary_is_blocked_by_boundaries():
    result = lookup("What is his salary expectation?")
    assert result.blocked
    assert result.top is not None
    assert result.top.kind == "boundaries"


def test_phone_number_is_blocked_by_boundaries():
    result = lookup("Can I get his phone number?")
    assert result.blocked
    assert result.top.kind == "boundaries"


def test_pending_project_is_answered_with_pending_block():
    result = lookup("Tell me about the CPU scheduling project")
    assert result.top is not None
    assert result.top.kind == "pending"


def test_deep_question_without_a_named_block_needs_retrieval():
    result = lookup("How did he handle the internal tradeoffs and why?")
    assert result.needs_depth
    assert not result.confident


def test_named_block_does_not_trigger_retrieval():
    result = lookup("Explain the Agentic Commerce Gateway")
    assert result.top is not None
    assert result.top.id == "project:agentic-commerce-gateway"
    assert not result.needs_depth


def test_unrelated_question_is_not_confident_and_offers_boundaries():
    result = lookup("What is the airspeed velocity of an unladen swallow?")
    assert not result.confident
    assert result.blocks
    assert result.top.kind == "boundaries"


def test_tokenize_strips_stopwords_and_plurals():
    assert tokenize("What are the projects?") == ["project"]


def test_generic_project_question_returns_a_confident_overview():
    result = lookup("What do you build?")
    assert result.confident
    assert result.reason == "generic project overview"
    assert any(block.kind == "project" for block in result.blocks)
    assert len(result.blocks) > 1


def test_current_work_routes_to_the_owner_block():
    result = lookup("What are you currently working on?")
    assert result.confident
    assert result.top is not None
    assert result.top.kind == "owner"


def test_payload_is_json_safe():
    payload = lookup("Tell me about QA-Forge").to_payload()
    assert payload["ids"]
    assert isinstance(payload["scores"], dict)
    assert all(isinstance(value, float) for value in payload["scores"].values())
